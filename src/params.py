
langs = {
    'fr': ['fr'],
    'en': ['en'],
    'de': ['de'],
    'enes': ['en','es'],
    'defren': ['de','en','fr']
}
lang_order = ['en']
datapath = 'dataset'
num_workers = 1
is_cpu = False
nanfiller = ' - '

palette = {
    'en-1': '#5e0305',
    'en-2': '#d70303',
    'en-3': '#f89fa6',
    'es-1': '#0359c8',
    'es-2': '#52a0ff',
    'es-3': '#aed2fe',
    'de-1': '#76550c',
    'de-2': '#c88e1a',
    'de-3': '#f8d468'
}
for lang in lang_order:
    for ii, sonority in enumerate(['1: obstruents', '2: nasal/approximant', '3: vowel']):
        palette[sonority+nanfiller+lang] = palette[f'{lang}-{3-ii}']
    for ii, voicing in enumerate(['voiced','voiceless']):
        palette[voicing+nanfiller+lang] = palette[f'{lang}-{3-ii}']
palette['en'] = palette['en-2']
palette['es'] = palette['es-2']
palette['de'] = palette['de-2']
palette[nanfiller*2+'en'] = '#999999'
palette[nanfiller*2+'es'] = '#999999'
palette[nanfiller*2+'de'] = '#999999'
palette[nanfiller] = '#999999'
palette['transfer'] = '#999999'
