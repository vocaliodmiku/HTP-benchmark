import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm 
import os
import numpy as np 
import pandas as pd
from copy import deepcopy
import sys
 
class Trainer:
    def __init__(self, model, config, logger=None, use_ddp=False, rank=0, use_amp=False):
        self.model = model
        self.config = config
        self.logger = logger
        self.use_ddp = use_ddp
        self.rank = rank
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=self.use_amp)
            
        self.lr = config.pretraining_lr
        self.checkpoint_path = os.path.join(self.config.folder, "pretraining")
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.epoch = -1
        self.losses = ["words"]
        self.reset_metrics()
        self.df = None

    def reset_metrics(self):
        self.train_losses = [0 for i_loss in range(len(self.losses))]
        self.train_accuracies = [0 for i_loss in range(len(self.losses))]
    
    def load_checkpoint(self, i_checkpoint=None):
        if i_checkpoint is None:
            state_file = "model_state.pth"
        else:
            state_file = "model_state_{}.pth".format(i_checkpoint)
        
        # Get the actual model (unwrap DDP if necessary)
        model = self.model.module if self.use_ddp else self.model
        loaded = True
        if os.path.isfile(os.path.join(self.checkpoint_path, state_file)): 
            if model.is_cuda:
                if i_checkpoint is None:
                    model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, state_file)), strict=True)
                else:
                    model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, state_file)), strict=True)
            else:
                model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, state_file), map_location="cpu"), strict=True)
            if self.rank == 0:
                print("Loaded previous model from {}".format(os.path.join(self.checkpoint_path, state_file)))
        else:
            if self.rank == 0:
                print("No checkpoint found at {}".format(os.path.join(self.checkpoint_path, state_file)))
                print("No previous model; starting from scratch")
                loaded = False
        return loaded
    
    def save_checkpoint(self, i_checkpoint=None):
        # Only save on main process
        if self.use_ddp and self.rank != 0:
            return
        
        try:
            # Get the actual model (unwrap DDP if necessary)
            model = self.model.module if self.use_ddp else self.model
            
            if i_checkpoint is None:
                torch.save(model.state_dict(), os.path.join(self.checkpoint_path, "model_state.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(self.checkpoint_path, "model_state_{}.pth".format(i_checkpoint)))
        except:
            print("Could not save model")
            
    def log(self, results):
        # Only log on main process
        if self.use_ddp and self.rank != 0:
            return
        
        if self.df is None:
            self.df = pd.DataFrame(columns=[field for field in results])
        self.df.loc[len(self.df)] = results
        self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

    def generate_inputs_and_targets(self, batch, lang_input=1, phone_target=1, word_target=1):
        """
        lang_input
        0: no language input
        1: language input half of the time
        2: language input always
        phone_target
        0: no phoneme targets
        1: concatenate phoneme inventories of different languages to use as targets
        2: superset of phonemes (combine homologues across languages) as targets
        word_target
        0: no word targets
        1: concatenate vocabularies of different languages to use as targets
               y[0].    y[1],     y[2],    y[3],         y[4],    y[5],     y[6],    y[7], y[8]
        (x || y_lang, y_phoneme, y_word_embedding, y_homologe, y_speech, y_speaker, y_iword, y_mask, y_word)
        """
        x, y,lengths = batch
        
        # Get the actual model (unwrap DDP if necessary)
        model = self.model.module if self.use_ddp else self.model
        
        if model.is_cuda:
            torch.cuda.empty_cache()
            x = x.cuda()
            y = tuple([yy.cuda() for yy in y if isinstance(yy, torch.Tensor)])
            lengths = lengths.cuda()
 
        y_lang = deepcopy(y[0][:,0])
        # loss_mask = [1,1,0] # ['phonemes', 'words', 'homologes']
        loss_mask = {"phonemes": 1, "words": 1, "homologes": 0}
        if lang_input==0: # Do not provide language tags in the input
            y_lang_in = torch.ones_like(y_lang)
        elif lang_input==1: # Provide language tags only half of the time
            is_langin = torch.rand(y_lang.shape)>.5
            y_lang_in = torch.ones_like(y_lang)*3
            y_lang_in[is_langin] = y_lang[is_langin]
        elif lang_input==2: # Always provide language tags
            y_lang_in = y_lang
        
        # Get the actual model (unwrap DDP if necessary)
        model = self.model.module if self.use_ddp else self.model
        
        if model.is_cuda:
            y_lang_in = y_lang_in.cuda()
             
        if phone_target==0:
            y[3].fill_(0)
            y[1].fill_(0)
            loss_mask["phonemes"]=0 # turn off phoneme loss
        elif phone_target==1:
            y[3].fill_(0)
        elif phone_target==2:
            y[1].fill_(0)
            
        if word_target==0:
            y[2].fill_(0)
        # targets = [y[1], y[2], y[3], y[8]] # phonemes, words, homologes, word_labels
        targets = {
            "phonemes": y[1],
            "words": y[2],
            "homologes": y[3],
            "word_labels": y[8]
        }
        
        inputs = [x, y_lang_in]
        mask = y[7]
        mask[mask==-1] = 0 # patch for the -1 padding in the mask
        return inputs, targets, lengths.long(), mask, loss_mask, y_lang

    def train(self, dataset, print_interval=100, tqdmout=sys.stderr, logger=None, calculate_accuracy=False):
        train_losses = {key: 0 for key in self.losses}
        train_accuracies = {key: 0 for key in self.losses}
        num_examples = 0
        self.model.train()
        model = self.model.module if self.use_ddp else self.model
        for idx, batch in enumerate(tqdm(dataset.loader,file=tqdmout)):
            inputs, targets, lengths, mask, loss_mask, lang_mask = self.generate_inputs_and_targets(
                batch, 
                self.config.pretraining_langin, 
                self.config.pretraining_phoneout,
                self.config.pretraining_wordout
            )
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs, lengths)
                loss, losses, accuracies = model.criterion(
                    inputs=outputs, 
                    targets=targets, 
                    lengths=lengths,
                    mask=mask, 
                    word_embedding=dataset.word_embedding_tensor,
                    loss_mask=loss_mask,
                    lang_mask=lang_mask,
                    phoneme_lang_mask=dataset.phoneme_lang_mask,
                    calculate_accuracy=calculate_accuracy,
                    loss_type=self.config.loss_type
                ) 
        
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            batch_size = inputs[0].shape[0]
            num_examples += batch_size
            for i_loss in self.losses:
                if losses[i_loss] is None: continue
                train_losses[i_loss] += losses[i_loss].cpu().data.numpy().item() * batch_size
                train_accuracies[i_loss] += accuracies[i_loss] * batch_size
 
        # WandB logging
        if logger is not None and calculate_accuracy:
            for i_loss in self.losses:
                logger.log({
                    f"train/batch_{i_loss}_loss": train_losses[i_loss]/num_examples,
                    f"train/batch_{i_loss}_accuracy": train_accuracies[i_loss]/num_examples,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch": self.epoch
                })
        
        for i_loss in self.losses:
            train_losses[i_loss] = train_losses[i_loss]/num_examples
            results = {"train_loss" : train_losses[i_loss], "set": 'train'}
            self.log(results)
            
        self.epoch += 1
        return train_losses

   
    def test(self, dataset, set='valid', h0=None, tqdmout=sys.stderr, logger=None):
        test_losses = {key: 0 for key in self.losses}
        test_accuracy = {key: 0 for key in self.losses}
        num_samples = 0
        num_examples = 0
 
        self.model.eval()
        model = self.model.module if self.use_ddp else self.model
        for idx, batch in enumerate(tqdm(dataset.loader,file=tqdmout)):
            inputs, targets, lengths, mask, loss_mask, lang_mask = self.generate_inputs_and_targets(
                batch, 
                self.config.pretraining_langin_test, 
                self.config.pretraining_phoneout,
                self.config.pretraining_wordout,
            )
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs, lengths, h0=h0,  training=False)
                loss, losses, accuracies = model.criterion(
                    inputs=outputs, 
                    targets=targets, 
                    lengths=lengths,
                    mask=mask, 
                    word_embedding=dataset.word_embedding_tensor,
                    loss_mask=loss_mask,
                    lang_mask=lang_mask,
                    phoneme_lang_mask=dataset.phoneme_lang_mask,
                    calculate_accuracy=True,
                    loss_type=self.config.loss_type
                )
            batch_size = len(lengths)
            num_examples += batch_size
            num_samples += int(sum(lengths))
            for i_loss in self.losses:
                if losses[i_loss] is None: continue
                test_losses[i_loss] += losses[i_loss].cpu().data.numpy().item() * batch_size
                test_accuracy[i_loss] += accuracies[i_loss].cpu().data.numpy().item() * batch_size
 
        if logger is not None:
            for i_loss in self.losses:
                if test_losses[i_loss] is None: continue
                logger.log({
                    f"{set}/batch_{i_loss}_loss": test_losses[i_loss]/num_examples,
                    f"{set}/batch_{i_loss}_accuracy": test_accuracy[i_loss]/num_examples,
                    f"epoch": self.epoch
                })
        
        for i_loss in self.losses:
            test_losses[i_loss] = test_losses[i_loss]/num_examples
            test_accuracy[i_loss] = test_accuracy[i_loss]/num_examples
            results = {"train_loss" : test_losses[i_loss], "set": set}
            self.log(results)
        return test_losses, test_accuracy
