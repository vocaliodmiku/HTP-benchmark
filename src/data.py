
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import configparser
import os
from subprocess import call
import soundfile as sf
import numpy as np
import textgrid
from audio import spectrogram
from tqdm import tqdm
import pandas as pd
import torch
import torch.utils.data
import gensim
from params import lang_order


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, config):
        """
        wav_paths: list of strings (wav file paths)
        config: Config object (contains info about model and training)
        """
        self.wav_paths = wav_paths # list of wav file paths
        self.length_mean = config.pretraining_length_mean
        self.length_var = config.pretraining_length_var

        # self.loader = torch.utils.data.DataLoader(self, batch_size=config.pretraining_batch_size, shuffle=True, collate_fn=CollateWavs())

    def __len__(self):
        return len(self.wav_paths)

    def read_audio(self, idx):
        x, fs = sf.read(self.wav_paths[idx])
        if x.ndim>1:
            return x.sum(1), fs
        else:
            return x, fs

    def __getitem__(self, idx):
        x, fs = self.read_audio(idx)
        return x, self.wav_paths[idx]

class ASRDataset(AudioDataset):
    def __init__(self, 
                 wav_paths, 
                 lang_ind, 
                 textgrid_paths, 
                 Sy_phoneme, 
                 Sy_word, 
                 Sy_homologe, 
                 config, train=True, num_samples=None, num_workers=8, use_ddp=False):
        """
        wav_paths: list of strings (wav file paths)
        textgrid_paths: list of strings (textgrid for each wav file)
        lang_ind: list of integers (language index for each wav file)
        Sy_phoneme: list of strings (all possible phonemes)
        Sy_word: list of strings (all possible words)
        config: Config object (contains info about model and training)
        use_ddp: whether to use DistributedSampler for DDP training
        """
        super(ASRDataset,self).__init__(wav_paths, config)

        self.textgrid_paths = textgrid_paths # list of textgrid file paths

        languages = np.array(lang_order)
        self.languages = languages
        self.lang_ind = np.zeros(len(lang_ind), dtype=int)
        for i_lang, language in enumerate(languages):
            self.lang_ind[lang_ind==language] = i_lang
        self.lang_sil = 0
        # self.Sy_phoneme = np.array([np.array(Sy) for Sy in Sy_phoneme])
        self.Sy_phoneme = Sy_phoneme
        self.phonemes = np.concatenate(Sy_phoneme)
        self.phoneme_lang_mask = torch.zeros((len(languages), len(self.phonemes)+1), dtype=int) # 1 for sil
        onset = 0
        for lang_idx, i in enumerate(Sy_phoneme): 
            self.phoneme_lang_mask[lang_idx, onset:onset+len(i)] = 1
            onset += len(i)
        
        self.Sy_homologe = Sy_homologe
        self.ind_nospeech = np.where((self.phonemes=='sil')|(self.phonemes=='sp')|(self.phonemes=='spn'))[0]
        if len(self.ind_nospeech)==0:
            self.ind_nospeech = np.array([-1,-1,-1])
        self.num_phonemes = sum([len(phones) for phones in Sy_phoneme])

        self.Sy_word = Sy_word
        self.words = np.concatenate(Sy_word)
        self.num_words = sum([len(words) for words in Sy_word])
        self.downsample_factor = config.downsample_factor
        speakers = np.array([wav_path.split('/')[-1][:10] for wav_path in wav_paths])
        self.speakers = np.zeros(len(speakers),dtype=int)
        
        self.word_embedding_ = gensim.models.KeyedVectors.load_word2vec_format(
            config.pretrained_word_embeddings_path)
        self.word_embedding = np.zeros((len(self.words), config.pretrained_word_embeddings_dim))
        num_words_assigned = 0
        offset = 0
        for lang, word_list in zip(languages, Sy_word):
            for i_word, word in enumerate(word_list):
                if word + "-" + lang in self.word_embedding_.key_to_index:
                    self.word_embedding[offset + i_word,:] = self.word_embedding_.get_vector(word + "-" + lang)
                    num_words_assigned += 1
            offset += len(word_list)
        print(f"Number of words assigned embeddings: {num_words_assigned} out of {len(self.words)}")
        
        # s = "-".join([str(sum(self.word_embedding[self.Sy_word[0].index(word)+len(self.Sy_word[0])])) for word in ["highlight", "uncle", "yellow"]])
        # print(f"Embedding check:{s}")
        
        self.word_embedding_dim = config.pretrained_word_embeddings_dim
        self.word_embedding_tensor = torch.tensor(self.word_embedding).float() # for cosine similarity calculations 
        
        self.oov_index = self.words.tolist().index("") if "" in self.words.tolist() else -1
        self.pattern_dict = {}
        
        self.prefetch = True
        self.delay200 = True
        if self.prefetch:
            self.prefetch_all() 
        else:
            pass
        
        if train:
            if use_ddp:
                # Use DistributedSampler for DDP training
                sampler = DistributedSampler(self, shuffle=True)
                self.loader = torch.utils.data.DataLoader(
                    self, 
                    batch_size=config.pretraining_batch_size,
                    sampler=sampler,
                    pin_memory=True,
                    num_workers=num_workers,
                    collate_fn=CollateItems()
                )
            else:
                self.loader = torch.utils.data.DataLoader(
                    self, 
                    batch_size=config.pretraining_batch_size, 
                    pin_memory=True,
                    num_workers=num_workers, 
                    shuffle=True, 
                    collate_fn=CollateItems()) # shuffle off because using sampler
                    # sampler=ImbalancedDatasetSampler(self, num_samples=num_samples))
        else:
            for i_speaker, speaker in enumerate(tqdm(np.unique(speakers))):
                self.speakers[speakers==speaker] = i_speaker
            self.loader = torch.utils.data.DataLoader(
                self, 
                batch_size=256, 
                pin_memory=True,
                num_workers=num_workers, 
                shuffle=False, 
                collate_fn=CollateItems())

    def get_feature(self, sig, sr): 
        spectrogram_Array = np.transpose(
            spectrogram(
                sig,        
                frame_shift_ms=10, 
                frame_length_ms=10, 
                sample_rate=sr
            )
        )
        return spectrogram_Array
    
    def prefetch_all(self):
        for idx in tqdm(range(len(self))):
            self.pattern_dict[idx] = self.__getitem__(idx)
        
    def __getitem__(self, idx):
        if self.prefetch:
            if idx in self.pattern_dict:
                return self.pattern_dict[idx]
        else:
            pass
        
        try:
            x, fs = self.read_audio(idx)
            feat = self.get_feature(x, fs)
            
        except Exception as e:
            print(f"Error loading {self.wav_paths[idx]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

        y_lang = self.lang_ind[idx]
        phn_index_offset = sum([len(ph) for ph in self.Sy_phoneme[:y_lang]])
        if os.path.isfile(self.textgrid_paths[idx]):
            tg = textgrid.TextGrid()
            tg.read(self.textgrid_paths[idx])

            y_phoneme, y_homologe = [], []
            for phoneme in tg.getList("phones")[0]:
                duration = phoneme.maxTime - phoneme.minTime
                phoneme = phoneme.mark.rstrip("0123456789")
                if phoneme in ['sil', 'sp', 'spn']:
                        
                    phoneme_index = -1 # self.Sy_phoneme[self.lang_sil].index(phoneme) + index_offset
                    homologe_index = -1 # self.Sy_homologe[self.lang_sil][phoneme]
                else:
                    if phoneme in self.Sy_phoneme[y_lang]: 
                        phoneme_index = self.Sy_phoneme[y_lang].index(phoneme) + phn_index_offset
                        homologe_index = self.Sy_homologe[y_lang][phoneme]
                    else:
                        phoneme_index = -1
                        homologe_index = -1
                if phoneme == '': phoneme_index = -1
                y_phoneme += [phoneme_index] * round(duration * fs)
                y_homologe += [homologe_index] * round(duration * fs)

            y_word = []
            y_mask = []
            y_iword = []
            word_index_offset = sum([len(ph) for ph in self.Sy_word[:y_lang]])
            for i_word, word in enumerate(tg.getList("words")[0]):
                duration = word.maxTime - word.minTime
                word_index = self.Sy_word[y_lang].index(word.mark)+word_index_offset if word.mark in self.Sy_word[y_lang] else self.oov_index
                    
                y_mask += [0 if word_index == self.oov_index else 1] * round(duration * fs)
                y_word += [word_index]* round(duration * fs)
                y_iword += [i_word] * round(duration * fs) # word count cumulative

        else:
            y_phoneme = np.ones(len(x)) * -1
            y_word = np.ones(len(x)) * self.oov_index
            y_iword = np.zeros(len(x))
            y_mask = np.zeros(len(x))
            print(f"Textgrid file {self.textgrid_paths[idx]} not found!")
            
        # Cut a snippet of length random_length from the audio
        snippet = [0, len(x)]
        if snippet is None:
            if self.length_mean>.5:
                random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.5))
            else:
                random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.05))
            if len(x) <= random_length:
                start = 0
                end = len(x)
            else:
                start = torch.randint(low=0, high=len(x)-random_length, size=(1,)).item()
                end = start + random_length
        else:
            start = snippet[0]
            end = snippet[1]

        # (I convert everything to numpy arrays, since there's a memory leak otherwise)
        x = x[start:end]
        # x[start_noise:end_noise] = np.random.randn(noise_len)/10
        y_phoneme = np.array(y_phoneme[start:end:self.downsample_factor])
        y_homologe = np.array(y_homologe[start:end:self.downsample_factor])
        y_word = y_word[start:end:self.downsample_factor]
        y_word_embedding = np.array([self.word_embedding[word] for word in y_word]) # get word embeddings and OOV words as zeros
        y_word = np.array(y_word)
        y_lang = y_lang.repeat(len(y_phoneme))
        y_speech = np.ones(y_lang.shape)
        y_speech[(self.ind_nospeech[0]==y_phoneme)|(self.ind_nospeech[1]==y_phoneme)|(self.ind_nospeech[2]==y_phoneme)] = 0
        # y_speaker = self.speakers[idx].repeat(len(y_phoneme))
        y_speaker = np.array([idx]).repeat(len(y_phoneme))
        y_iword = np.array(y_iword[start:end:self.downsample_factor])
        y_mask = np.array(y_mask[start:end:self.downsample_factor])

        if feat.shape[0]!=len(y_word):
            min_len = min(feat.shape[0], len(y_word))
            feat = feat[:min_len,:]
            y_lang = y_lang[:min_len]
            y_phoneme = y_phoneme[:min_len]
            y_word_embedding = y_word_embedding[:min_len,:]
            y_homologe = y_homologe[:min_len]
            y_speech = y_speech[:min_len]
            y_speaker = y_speaker[:min_len]
            y_iword = y_iword[:min_len]
            y_mask = y_mask[:min_len]
            y_word = y_word[:min_len]
        
        # PATCH: mask the first 5 frames 
        y_mask[:10] = 0
        
        # Add 200ms of silence at the beginning the audio
        if self.delay200:
            feat = np.pad(feat, ((20, 0), (0, 0)), mode='constant') # 20 frames of 10ms each = 200ms
            y_lang = np.pad(y_lang, (20, 0), mode='constant', constant_values=-1)
            y_phoneme = np.pad(y_phoneme, (20, 0), mode='constant', constant_values=-1)
            y_word_embedding = np.pad(y_word_embedding, ((20, 0), (0, 0)), mode='constant')
            y_homologe = np.pad(y_homologe, (20, 0), mode='constant', constant_values=-1)
            y_speech = np.pad(y_speech, (20, 0), mode='constant', constant_values=0)
            y_speaker = np.pad(y_speaker, (20, 0), mode='constant', constant_values=-1)
            y_iword = np.pad(y_iword, (20, 0), mode='constant', constant_values=-1)
            y_mask = np.pad(y_mask, (20, 0), mode='constant', constant_values=0)
            y_word = np.pad(y_word, (20, 0), mode='constant', constant_values=self.oov_index)
        
        return feat, y_lang, y_phoneme, y_word_embedding, y_homologe, y_speech, y_speaker, y_iword, y_mask, y_word



class CollateItems:
    def __call__(self, batch):
        """
        batch: list of tuples (input wav, phoneme labels, word labels)
        Returns a minibatch of wavs and labels as Tensors.
        """

        n_items = len(batch[0])
        batch_size = len(batch)
        x = []
        ys = [[] for _ in range(n_items-1)]
        lengths = []
        for index in range(batch_size):
            # for each item in the batch
            items_ = batch[index]
            x.append(torch.tensor(items_[0]).float()) # input feat
            for item_, y in zip(items_[1:], ys):
                y.append( torch.tensor(item_).float() if item_.ndim==2 else torch.tensor(item_).long() )
            lengths.append(torch.tensor(items_[0].shape[0]).long())

        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True) 
        for i_y in range(len(ys)):
            if ys[i_y][0].ndim==2: # word embeddings and feat
                ys[i_y] = torch.nn.utils.rnn.pad_sequence(ys[i_y], batch_first=True)
            else:
                ys[i_y] = torch.nn.utils.rnn.pad_sequence(ys[i_y], padding_value=-1).T
        lengths = torch.stack(lengths)

        return (x, tuple(ys), lengths)

class Config:
    def __init__(self):
        self.thats_me = True

def read_config(config_file, name):
    config = Config()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    #[experiment]
    config.seed=int(parser.get("experiment", "seed"))
    config.folder = os.path.join(os.path.splitext(config_file)[0], name)
    # config.folder=parser.get("experiment", "folder")

    # Make a folder containing experiment information
    if not os.path.isdir(config.folder):
        os.makedirs(config.folder)
        os.makedirs(os.path.join(config.folder, "pretraining"))
        os.makedirs(os.path.join(config.folder, "training"))
    call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

    #[model]
    config.type=parser.get("model", "type")
    config.n_mel=int(parser.get("model", "n_mel"))
    config.n_mfcc_out=int(parser.get("model", "n_mfcc_out"))
    config.inp_size=int(parser.get("model", "inp_size"))
    config.rnn_hidden_size=int(parser.get("model", "rnn_hidden_size"))
    config.num_rnn_layers=int(parser.get("model", "num_rnn_layers"))
    config.rnn_drop=float(parser.get("model", "rnn_drop"))
    config.rnn_bidirectional=parser.get("model", "rnn_bidirectional") == "True"
    config.downsample_factor=int(parser.get("model", "downsample_factor"))
    config.vocabulary_size=int(parser.get("model", "vocabulary_size"))
    config.pretrained_word_embeddings_path=parser.get("model", "pretrained_word_embeddings_path")
    config.pretrained_word_embeddings_dim=int(parser.get("model", "pretrained_word_embeddings_dim"))
    config.cnn_modules=int(parser.get("model", "cnn_modules"))
    config.loss_type=parser.get("model", "loss_type")
    
    #[pretraining]
    config.pretraining_manifest_train=parser.get("pretraining", "pretraining_manifest_train")
    config.pretraining_manifest_dev=parser.get("pretraining", "pretraining_manifest_dev")
    config.pretraining_manifest_test=parser.get("pretraining", "pretraining_manifest_test")
    config.fs=int(parser.get("pretraining", "fs"))
    config.time_shift=int(parser.get("pretraining", "time_shift"))
    config.n_output_quantize=int(parser.get("pretraining", "n_output_quantize"))
    config.pretraining_lr=float(parser.get("pretraining", "pretraining_lr"))
    config.pretraining_patience=float(parser.get("pretraining", "pretraining_patience"))
    config.pretraining_lr_factor=float(parser.get("pretraining", "pretraining_lr_factor"))
    config.pretraining_batch_size=int(parser.get("pretraining", "pretraining_batch_size"))
    config.pretraining_num_epochs=int(parser.get("pretraining", "pretraining_num_epochs"))
    config.pretraining_length_mean=float(parser.get("pretraining", "pretraining_length_mean"))
    config.pretraining_length_var=float(parser.get("pretraining", "pretraining_length_var"))
    config.grad_clip=float(parser.get("pretraining", "grad_clip"))
    config.pretraining_phoneout=int(parser.get("pretraining", "pretraining_phoneout"))
    config.pretraining_wordout=int(parser.get("pretraining", "pretraining_wordout"))
    config.pretraining_langin=int(parser.get("pretraining", "pretraining_langin"))
    config.pretraining_langin_test=int(parser.get("pretraining", "pretraining_langin_test"))
    config.pretraining_eval_interval=int(parser.get("pretraining", "pretraining_eval_interval"))

    #[training]
    config.training_manifest_train=parser.get("training", "training_manifest_train")
    config.training_manifest_dev=parser.get("training", "training_manifest_dev")
    config.training_manifest_test=parser.get("training", "training_manifest_test")
    config.training_type=parser.get("training", "training_type")
    config.training_lr=float(parser.get("training", "training_lr"))
    config.training_patience=int(parser.get("training", "training_patience"))
    config.training_batch_size=int(parser.get("training", "training_batch_size"))
    config.training_num_epochs=int(parser.get("training", "training_num_epochs"))
    config.training_pretrained_out_layer=parser.get("training", "training_pretrained_out_layer")
    config.training_inp_dim=int(parser.get("training", "training_inp_dim"))
    return config

def load_manifest(manifest_path, datapath):
    with open(os.path.join(datapath, manifest_path)) as f:
        wavfiles, tgfiles, languages = ([], [], [])
        for line in f.readlines():
            filename = line.strip('\n').split("\t")[0]
            languages.append(filename.split('/')[0])
            filepath = os.path.join(datapath, filename)
            wavfiles.append(filepath+'.wav')
            tgfiles.append(filepath+'.TextGrid')
    return np.array(wavfiles), np.array(tgfiles), np.array(languages)

def get_datasets(config, datapath, manifest_train, manifest_dev, manifest_test, num_workers, folder='pretraining', use_ddp=False):
    wavfiles, tgfiles, languages = load_manifest(manifest_train, datapath)
    tgfiles_vocab = tgfiles
    languages_vocab = languages
    Sy_phoneme, Sy_word, Sy_homologe = ([], [], [])
    for i_lang, language in enumerate(lang_order):
        df = pd.read_csv(f'{datapath}/phonemes_{language}.csv',index_col=0,na_filter=False)
        Sy_phoneme.append(df.index.values.tolist())
        Sy_homologe.append(df['homolog'].values.tolist())
        df = pd.read_csv(f'{datapath}/words_{language}.csv',index_col=0,na_filter=False)
        Sy_word.append(df['word'].values.tolist())
    
    # Create a list of unique homologes across all languages
    homologes = np.unique(np.concatenate(Sy_homologe)).tolist()
    for i_lang in range(len(lang_order)):
        # Map the homologe strings to their unique indices
        Sy_homologe[i_lang] = {ph: homologes.index(hom) for hom, ph in zip(Sy_homologe[i_lang],Sy_phoneme[i_lang])}
        
    config.num_phonemes = sum([len(phones) for phones in Sy_phoneme])
    config.num_homologes = len(homologes)
    config.vocabulary_size = sum([len(words) for words in Sy_word])
    config.languages = np.unique(languages_vocab)
    
    # Prepare training dataset
    wavfiles, tgfiles, languages = load_manifest(manifest_train, datapath)
    num_samples = len(wavfiles)
    train_dataset = ASRDataset(
        wav_paths=wavfiles, 
        lang_ind=languages, 
        textgrid_paths=tgfiles, 
        Sy_phoneme=Sy_phoneme,
        Sy_word=Sy_word,
        Sy_homologe=Sy_homologe,
        config=config,
        train=True, 
        num_samples=num_samples, 
        num_workers=num_workers,
        use_ddp=use_ddp
    )
    pretraining_length_mean = config.pretraining_length_mean
    pretraining_length_var = config.pretraining_length_var
    pretraining_batch_size = config.pretraining_batch_size
    config.pretraining_length_mean = 100 # Use all
    config.pretraining_length_var = 0
    config.pretraining_batch_size = 4
    
    # Prepare validation and test datasets
    wavfiles, tgfiles, languages = load_manifest(manifest_dev, datapath)
    valid_dataset = ASRDataset(
        wav_paths=wavfiles, 
        lang_ind=languages, 
        textgrid_paths=tgfiles, 
        Sy_phoneme=Sy_phoneme, 
        Sy_word=Sy_word, 
        Sy_homologe=Sy_homologe,
        config=config, 
        train=False, 
        num_workers=num_workers)
    
    wavfiles, tgfiles, languages = load_manifest(manifest_test, datapath)
    test_dataset = ASRDataset(wavfiles, languages, tgfiles, Sy_phoneme, Sy_word, Sy_homologe,
                              config, False, num_workers=num_workers)
    config.pretraining_length_mean = pretraining_length_mean
    config.pretraining_length_var = pretraining_length_var
    config.pretraining_batch_size = pretraining_batch_size

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    config = read_config('experiments/enes_words_srv.cfg', "test")
    datapath = 'dataset'
    train_dataset, valid_dataset, test_dataset = get_datasets(config, datapath,
        config.pretraining_manifest_train,
        config.pretraining_manifest_dev,
        config.pretraining_manifest_test, num_workers=1)
    
    for idx, batch in enumerate(tqdm(train_dataset.loader)):
        pass
    batch = [train_dataset[0], train_dataset[1], train_dataset[2]]
    x, y, lengths = CollateItems()(batch)