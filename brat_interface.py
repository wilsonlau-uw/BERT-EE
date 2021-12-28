import pickle
import os
from os import listdir
from os.path import isfile, join, splitext
from shutil import copyfile
from tqdm import tqdm
from collections import Counter,defaultdict
import random
import numpy as np
import pandas as pd
import re
import spacy

from spacy.tokenizer import Tokenizer
import itertools
from config import Config

nlp = spacy.load(Config.getstr('general','spacy_model'))
nlp.max_length = 10000000

from logger import Logger
from multiprocessing import Pool
from string import punctuation

class BRAT_interface():

    def __init__(self  ):

        self._use_spacy = Config.getboolean('general','use_spacy_tokenizer') == True

        if self._use_spacy:
            self.tokenize=self.tokenize_spacy
        else:
            self.tokenize = self.tokenize_split

    def merge_tokens(self,tokens):
        out_tokens=[]
        added_spaces=0
        if tokens[0][-1]==' ' :
            out_tokens.append(tokens[0])
        else :
            out_tokens.append(tokens[0]+' ')
            added_spaces+=1

        for i in range(1,len(tokens)):
            if len(tokens[i].strip())==0:
                out_tokens[-1]=out_tokens[-1]+tokens[i]
            elif tokens[i][-1]==' ' :
                out_tokens.append(tokens[i])
            else :
                out_tokens.append(tokens[i]+' ')
                added_spaces+=1

        return out_tokens,added_spaces


    def handle_span_with_subtokens(self,text, start_span, end_span, start_span_start_idx, start_span_end_idx, end_span_start_idx,end_span_end_idx,start_span_label,end_span_label):
        if start_span_start_idx < end_span_start_idx:
            if text[start_span_start_idx - 1].strip().__len__() > 0:
                Logger.debug('first span changed from "{}" to "{}"'.format(text[start_span_start_idx - 3:start_span_end_idx],
                                                                        text[
                                                                        start_span_start_idx - 3:start_span_start_idx] + ' ' + text[
                                                                                                                               start_span_start_idx:start_span_start_idx + len(
                                                                                                                                   start_span_label)]))
                text = text[0:start_span_start_idx] + ' ' + text[start_span_start_idx:]
                start_span_start_idx += 1
                start_span_end_idx += 1
                end_span_start_idx += 1
                end_span_end_idx += 1
                assert text[start_span_start_idx:start_span_end_idx] == start_span
                assert text[end_span_start_idx:end_span_end_idx] == end_span

            if text[end_span_start_idx - 1].strip().__len__() > 0:
                Logger.debug('second span changed from "{}" to "{}"'.format(text[end_span_start_idx - 3:end_span_end_idx],
                                                                         text[
                                                                         end_span_start_idx - 3:end_span_start_idx] + ' ' + text[
                                                                                                                            end_span_start_idx:end_span_start_idx + len(
                                                                                                                                end_span_label)]))
                text = text[0:end_span_start_idx] + ' ' + text[end_span_start_idx:]
                end_span_start_idx += 1
                end_span_end_idx += 1

                if start_span_end_idx > end_span_start_idx:
                    start_span_end_idx += 1

                assert text[start_span_start_idx:start_span_end_idx] == start_span
                assert text[end_span_start_idx:end_span_end_idx] == end_span

            if text[start_span_end_idx].strip().__len__() > 0:
                Logger.debug('first span changed from "{}" to "{}"'.format(text[start_span_start_idx:start_span_end_idx + 3],
                                                                        text[
                                                                        start_span_start_idx:start_span_end_idx] + ' ' + text[
                                                                                                                         start_span_end_idx:start_span_end_idx + 3]))
                text = text[0:start_span_end_idx] + ' ' + text[start_span_end_idx:]
                if end_span_start_idx > start_span_end_idx:
                    end_span_start_idx += 1
                    end_span_end_idx += 1

                assert text[start_span_start_idx:start_span_end_idx] == start_span
                assert text[end_span_start_idx:end_span_end_idx] == end_span

        elif start_span_start_idx > end_span_start_idx:
            if text[end_span_start_idx - 1].strip().__len__() > 0:
                Logger.debug('first span changed from "{}" to "{}"'.format(text[end_span_start_idx - 3:end_span_end_idx],
                                                                        text[
                                                                        end_span_start_idx - 3:end_span_start_idx] + ' ' + text[
                                                                                                                           end_span_start_idx:end_span_start_idx + len(
                                                                                                                               end_span_label)]))
                text = text[0:end_span_start_idx] + ' ' + text[end_span_start_idx:]
                start_span_start_idx += 1
                start_span_end_idx += 1
                end_span_start_idx += 1
                end_span_end_idx += 1
                assert text[start_span_start_idx:start_span_end_idx] == start_span
                assert text[end_span_start_idx:end_span_end_idx] == end_span

            if text[start_span_start_idx - 1].strip().__len__() > 0:
                Logger.debug('second span changed from "{}" to "{}"'.format(text[start_span_start_idx - 3:start_span_end_idx],
                                                                         text[
                                                                         start_span_start_idx - 3:start_span_start_idx] + ' ' + text[
                                                                                                                                start_span_start_idx:start_span_start_idx + len(
                                                                                                                                    start_span_label)]))
                text = text[0:start_span_start_idx] + ' ' + text[start_span_start_idx:]
                start_span_start_idx += 1
                start_span_end_idx += 1

                assert text[start_span_start_idx:start_span_end_idx] == start_span
                assert text[end_span_start_idx:end_span_end_idx] == end_span

            if text[end_span_end_idx].strip().__len__() > 0:
                Logger.debug('first span changed from "{}" to "{}"'.format(text[end_span_start_idx:end_span_end_idx + 3],
                                                                        text[
                                                                        end_span_start_idx:end_span_end_idx] + ' ' + text[
                                                                                                                     end_span_end_idx:end_span_end_idx + 3]))
                text = text[0:end_span_end_idx] + ' ' + text[end_span_end_idx:]
                if start_span_start_idx > end_span_end_idx:
                    start_span_start_idx += 1
                if start_span_end_idx > end_span_end_idx:
                    start_span_end_idx += 1
                assert text[start_span_start_idx:start_span_end_idx] == start_span
                assert text[end_span_start_idx:end_span_end_idx] == end_span

        return text, start_span_start_idx, start_span_end_idx, end_span_start_idx,end_span_end_idx



    def normalize_unlabelled(self, samples, txt, fname, tokens, tags,use_sent_segmentation=False):

        if use_sent_segmentation:
            sents = [s.text for s in nlp(txt).doc.sents]
        else:
            sents = txt.split('\n')

        start_idx=0
        token_s=0
        parse_txt=txt
        for s in sents:

            sent_tokens = self.tokenize(s)

            end_idx = start_idx + len(sent_tokens )
            if len(tokens[start_idx:end_idx])>0:
                for t in tokens[start_idx:end_idx]:
                    token = t[0]

                    index= parse_txt.index(token) +token_s

                    assert t[1] is None or index ==t[1]
                    if t[1] is None:
                        t[1] = index
                    token_s = index+token.__len__()
                    parse_txt=txt[token_s:]


                samples.append({'filename': fname, 'tokens': [t[0] for t in tokens[start_idx:end_idx]],
                                'indices': [t[1] for t in tokens[start_idx:end_idx]], 'labels': tags[start_idx:end_idx]})

            start_idx=end_idx


    def normalize(self,samples, txt, fname, tokens, tags,use_sent_segmentation=False):

        if use_sent_segmentation:
            sents = [s.text for s in nlp(txt).doc.sents]
        else:
            sents = txt.split('\n')

        start_idx=0
        token_s=0
        parse_txt=txt
        for s in sents:

            end_idx = start_idx+len(s.split())
            if len(tokens[start_idx:end_idx])>0:
                # if all([t=='O' for t in tags[start_idx:end_idx]]):
                for t in tokens[start_idx:end_idx]:
                    token = t[0]

                    index= parse_txt.index(token) +token_s

                    assert t[1] is None or index ==t[1]
                    if t[1] is None:
                        t[1] = index
                    token_s = index+token.__len__()
                    parse_txt=txt[token_s:]

            start_idx=end_idx

        start_idx=0
        end_idx=0
        for s in sents:
            found = True

            if s.strip().__len__()==0: continue
            end_idx += 1

            while not ' '.join([t[0] for t in tokens[start_idx:end_idx]]).endswith(' '.join(s.split())) :
                if tokens[start_idx:end_idx].__len__() > s.split().__len__() or end_idx > len(tokens):
                    found=False
                    end_idx -= 1
                    break
                end_idx += 1

            if found:
                tokensized = tokens[start_idx:end_idx]
                samples.append({'filename': fname, 'tokens': [t[0] for t in tokensized],
                                'indices': [t[1] for t in tokensized], 'labels': tags[start_idx:end_idx]})


                assert  ' '.join([t[0] for t in tokensized]).endswith(' '.join(s.split()))

                start_idx = end_idx
            else:

                end_idx -= 1
                start_idx = end_idx


    def normalize0(self,samples, txt, fname, tokens, tags,use_sent_segmentation=False):

        if use_sent_segmentation:
            sents = [s.text for s in nlp(txt).doc.sents]
        else:
            sents = txt.split('\n')

        start_idx=0
        token_s=0
        parse_txt=txt
        for s in sents:

            end_idx = start_idx+len(s.split())
            if len(tokens[start_idx:end_idx])>0:
                # if all([t=='O' for t in tags[start_idx:end_idx]]):
                for t in tokens[start_idx:end_idx]:
                    token = t[0]

                    index= parse_txt.index(token) +token_s

                    assert t[1] is None or index ==t[1]
                    if t[1] is None:
                        t[1] = index
                    token_s = index+token.__len__()
                    parse_txt=txt[token_s:]

            start_idx=end_idx

        start_idx=0
        end_idx=0
        for s in sents:
            found = True

            if s.strip().__len__()==0: continue
            end_idx += 1

            while not ' '.join([t[0] for t in tokens[start_idx:end_idx]]).endswith(' '.join(s.split())) :
                if tokens[start_idx:end_idx].__len__() > s.split().__len__() or end_idx > len(tokens):
                    found=False
                    end_idx -= 1
                    break
                end_idx += 1

            if found:
                tokensized = tokens[start_idx:end_idx]
                samples.append({'filename': fname, 'tokens': [t[0] for t in tokensized],
                                'indices': [t[1] for t in tokensized], 'labels': tags[start_idx:end_idx]})

                assert  ' '.join([t[0] for t in tokensized]).endswith(' '.join(s.split()))

                start_idx = end_idx
            else:
                end_idx -= 1
                start_idx = end_idx



    def replace_punctuation(self,text):
        punctuation_list =  ['!','"','’','$','&','(',')',',',':',';','<','>','[',']','{','}','/','~','-','=']
        for p in punctuation_list:
            text=text.replace(p,' '+p+' ')

        return text

    def replace_unbalanced_punctuation(self,text):
        punctuation_list_start = ['(', '<', '[' ,'{' ]
        punctuation_list_end = [')', '>', ']' ,'}' ]
        for i,p in enumerate(punctuation_list_start):
            if text.startswith(p) and not text.endswith(punctuation_list_end[i]):
                text=p+' '+text[1:]
        for i,p in enumerate(punctuation_list_end):
            if text.endswith(p) and not text.startswith(punctuation_list_start[i]):
                text = text[:-1] + ' ' + p

        return text

    def replace_ending_punctuation(self,text):
        punctuation_list =  [',','.','!',':',';']
        for p in punctuation_list:
            if text.endswith(p):
                text=text[:-1]+' '+p

        return text

    def tokenize_spacy(self,text ):
        return [t.text  for t in nlp(' '.join(self.replace_punctuation(text).split())).doc]

    def tokenize_split(self,text ):
        tokens_list= [self.replace_ending_punctuation(self.replace_unbalanced_punctuation(t)).split() for t in text.split()]
        return list(itertools.chain.from_iterable(tokens_list))

    def read_T_indices(self,l):
        if (l.startswith('T')):
            toks = l.split('\t')[1]
            label = toks.split()[0]
            start = int(toks.split()[1])
            end = int(toks.split()[-1])
        return  start,end

    def read_brat_unlabelled_entities(self,brat_folder,use_sent_segmentation=False):

        Logger.debug('reading brat files in folder ', brat_folder)
        txt_files = [f for f in listdir(brat_folder) if isfile(join(brat_folder, f))and f.endswith('.txt')]
        sortedFiles = [join(brat_folder, splitext(a)[0]) for a in txt_files]
        sortedFiles.sort()
        sample_idx_offset=0
        for f, fname in enumerate(sortedFiles):
            samples = []

            Logger.info(fname)

            with open(fname + '.txt', newline='',mode='rU') as txt_file:
                out_txt = txt_file.read()

                tokens = [[t, None] for t in self.tokenize(out_txt) ]
                self.normalize_unlabelled(samples, out_txt, fname, tokens, [],use_sent_segmentation)
                for i, s in enumerate(samples):
                    s['sample_idx'] = i+sample_idx_offset

                sample_idx_offset +=samples.__len__()
                yield  samples ,out_txt

    def get_brat_files(self,brat_folder):
        if not os.path.isdir(brat_folder):
            Logger.error(brat_folder + ' is not a directory')
        elif [f for f in listdir(brat_folder) if isfile(join(brat_folder, f)) and f.endswith('.ann')].__len__() == 0:
            Logger.error('No BRAT annotation files in ' + brat_folder)

        ann_files = [f for f in listdir(brat_folder) if
                     isfile(join(brat_folder, f)) and f.endswith('.ann') and os.stat(join(brat_folder, f)).st_size > 0]
        txt_files = [f for f in listdir(brat_folder) if isfile(join(brat_folder, f)) and f.endswith('.txt') and os.stat(
            join(brat_folder, splitext(f)[0] + '.ann')).st_size > 0]
        assert len(txt_files) == len(ann_files)

        return ann_files, txt_files

    def get_brat_unlabelled_files(self,brat_folder):
        if not os.path.isdir(brat_folder):
            Logger.error(brat_folder + ' is not a directory')
        elif [f for f in listdir(brat_folder) if isfile(join(brat_folder, f)) and f.endswith('.txt')].__len__() == 0:
            Logger.error('No BRAT annotation files in ' + brat_folder)

        txt_files = [f for f in listdir(brat_folder) if isfile(join(brat_folder, f)) and f.endswith('.txt')]

        return txt_files


    def read_brat_unlabelled_events(self,brat_folder, event_entities,distance=None, max_sent_windows=None,no_relation_ratio=None, num_workers=1):
        Logger.debug('reading brat files in folder ', brat_folder)
        ann_files, txt_files = self.get_brat_files(brat_folder)
        sortedFiles = [join(brat_folder, splitext(a)[0]) for a in ann_files]
        sortedFiles.sort()
        for f, fname in enumerate(sortedFiles):
            event_files= self.brat_to_events(fname)
            yield self.events_to_dict(event_files, event_entities=event_entities,distance=distance,max_sent_windows=max_sent_windows, unlabelled=True,
                              no_relation_ratio=no_relation_ratio, num_workers=num_workers), fname


    def check_brat_ann(self,fname,txt,ann_line,last_indices, allow_overlap=False):
        anns = ann_line.split()
        label = anns[1]
        start_index,end_index=self.read_T_indices(ann_line)
        span = txt[start_index:end_index]

        if span.strip() != span:
            Logger.debug('({}.ann) line ({}). Span contains beginning/trailing space.  Ignoring.....'.format(fname,
                                                                                                                    ann_line[
                                                                                                                    :-1]))
            return False

        if txt[start_index:end_index] != span:
            Logger.debug('({}.ann) line ({}). Indices not matching span ( {} ) vs ( {} ).  Ignoring.....'.format(fname,
                                                                                                               ann_line[
                                                                                                               :-1],  txt[start_index:end_index] ,span))
            return False
        if not allow_overlap and start_index <last_indices[1]:
            Logger.debug('({}.ann) line ({}). overlapping span annotation.  Ignoring.....'.format(fname,
                                                                                                                    ann_line[
                                                                                                                    :-1]))
            return False
        if end_index - start_index != len(span):
            Logger.debug('({}.ann) line ({}). Indice length not matching span length.  Ignoring.....'.format(fname,
                                                                                                                    ann_line[
                                                                                                                    :-1]))
            return False

        else:
            last_indices[0]=start_index
            last_indices[1] =end_index
            return True

    def read_brat(self,brat_folder,labels=None, schema='IOB2',allow_overlap=False,use_sent_segmentation=False):

        assert labels==None or isinstance(labels,list)

        ann_files = [f for f in listdir(brat_folder) if isfile(join(brat_folder, f)) and f.endswith('.ann')]
        txt_files = [f for f in listdir(brat_folder) if isfile(join(brat_folder, f))and f.endswith('.txt')]
        samples = []
        labels_count={}

        assert len(txt_files) == len(ann_files)

        sortedFiles = [join(brat_folder, splitext(a)[0]) for a in txt_files]
        sortedFiles.sort()
        for f, fname in enumerate(tqdm(sortedFiles)):
            Logger.debug('reading file {}'.format(fname))
            tokens = []
            tags = []
            with open(fname+'.txt',newline='', mode='rU' ) as txt_file,  open(fname+'.ann', 'r') as ann_file:
                txt = txt_file.read()
                ann_lines = ann_file.readlines()
                ann_lines_attrs = [l for l in ann_lines if l.lstrip().startswith('A')]
                ann_lines = [l for l in ann_lines if l.lstrip().startswith('T')]
                ann_lines = sorted(ann_lines, key=lambda x: (self.read_T_indices(x)))

                ann_lines_nodups = []
                for a in ann_lines:
                    if (a[a.index(' ')+1:] not in [a[a.index(' ')+1:] for a in ann_lines_nodups]):
                        ann_lines_nodups.append(a)

                # merge entity labels with attributes
                ann_lines_attrs_map = dict([(a[2], a[3]) for a in [a.split() for a in ann_lines_attrs]])

                txt_idx =0
                out_txt = txt
                added_spaces=0
                start_index=0
                end_index=0
                last_indices = [0, 0]

                for ann_line in ann_lines:

                    anns = ann_line.split()
                    label = anns[1] #.lower()

                    if not self.check_brat_ann(fname,txt,ann_line,last_indices,allow_overlap):
                        continue

                    if ann_lines_attrs.__len__()>0 and anns[0] in ann_lines_attrs_map:
                        label = label+' ('+ann_lines_attrs_map[anns[0]]+')'

                    if labels != None and anns[1] not in [l.strip() for l in labels]:
                        continue

                    if label not in labels_count: labels_count[label]=0
                    labels_count[label]+=1

                    start_index,end_index=self.read_T_indices(ann_line)
                    start_index+=added_spaces
                    end_index+=added_spaces

                    # replace \n \t \xa0 with ' '
                    out_txt = out_txt[0:start_index] + out_txt[start_index:end_index].replace('\n',' ').replace('\t',' ').replace(u'\xa0', ' ') + out_txt[end_index:]

                    # in case annotation starts/ends at part of a token, insert a space in between to split the token
                    if start_index - 1 >= 0 and out_txt[start_index - 1].strip().__len__() > 0:
                        out_txt = out_txt[0:start_index] + ' ' + out_txt[start_index:]
                        added_spaces += 1
                        start_index+=1
                        end_index+=1

                    if end_index < len(out_txt) and out_txt[end_index].strip().__len__() > 0:
                        out_txt = out_txt[0:end_index] + ' ' + out_txt[end_index:]
                        added_spaces += 1

                    ann_tokens_with_ws, parsed_added_spaces = self.merge_tokens([ t.text_with_ws  for t in nlp(out_txt[start_index:end_index])])

                    ann_tokens = [t.strip() for t in ann_tokens_with_ws]
                    parsed_tokens_diff = ann_tokens.__len__() - out_txt[start_index:end_index].split().__len__()
                    if parsed_tokens_diff>0:
                        out_txt = out_txt[0:start_index] + ''.join(ann_tokens_with_ws) + out_txt[end_index:]
                        added_spaces+= parsed_added_spaces
                        end_index+=parsed_added_spaces

                    ann_tokens_idx = [start_index]
                    for i, t in enumerate(ann_tokens_with_ws):
                        if i>0:
                            ann_tokens_idx.append(ann_tokens_idx[i-1]+len(ann_tokens_with_ws[i-1]))


                    if(txt_idx < start_index):
                        txt_line = out_txt[txt_idx:start_index]

                        tokens_O = [a for a in txt_line.split() if a.strip()!='']
                        for t in tokens_O:
                            tokens.append([t,None])
                            tags.append('O')

                        txt_idx=end_index

                        if(ann_tokens.__len__()==1):
                            if schema=='IOBES':
                                tags.append('S-' + label)
                            else:  # IOB2
                                tags.append('B-' + label)

                            tokens.append([ann_tokens[0],ann_tokens_idx[0]])
                            assert out_txt[ann_tokens_idx[0]:ann_tokens_idx[0]+len(ann_tokens[0])]==ann_tokens[0]


                        else:
                            for i,t in enumerate(ann_tokens):
                                tokens.append([t,ann_tokens_idx[i]])
                                assert out_txt[ann_tokens_idx[i]:ann_tokens_idx[i] + len(t)] == t

                                if i==0:
                                    tag='B-'+label
                                elif i< ann_tokens.__len__()-1:
                                    tag = 'I-' + label
                                else:
                                    if schema == 'IOBES':
                                        tag = 'E-' + label
                                    else:  # IOB2
                                        tag = 'I-' + label

                                tags.append(tag)


                        assert len(tags) == len(tokens)


                remaining_tokens = out_txt[end_index:].split()

                if remaining_tokens.__len__()>0:
                    tokens.extend([[t, None] for t in remaining_tokens])
                    tags.extend(['O']*remaining_tokens.__len__())

                self.normalize(samples,out_txt,fname,tokens,tags,use_sent_segmentation)

        for i,s in enumerate(samples):
            s['sample_idx']=i

        for l in labels_count:
            Logger.info('# of {} : {}'.format(l, labels_count[l]))

        return samples


    def save_brat_ner(self,file, out_folder,outputs,  tokens,  tokens_indices, txt,schema='IOB2'):

        txtfile=file+'.txt'
        annfile = file + '.ann'
        copyfile(txtfile, os.path.join(out_folder,os.path.basename(txtfile)))

        outputs_list = list(itertools.chain.from_iterable(outputs))

        with open(os.path.join(out_folder, os.path.basename(annfile)), 'w') as wf:
            Tindex=1
            Eindex=1
            Aindex=1
            c=1
            current_tokens=[]
            for i, o in enumerate(outputs_list):

                if o!= 'O' and '-' in o :
                    label = o[o.strip().split('\t')[-1].index('-')+1:]
                elif o[0] == '[':
                    label = 'O'
                else:
                    label = o

                categorical = ' (' in label
                categorical_value=''
                if categorical:  # it is categorical
                    categorical_value = label.split(' (')[1].rstrip(')')
                    label = label.split(' (')[0]

                tag = o[0]
                token = tokens[i]

                if schema=='IOBES': # to do

                    def writeToken(Tindex,Eindex,Aindex,label,  start_idx, end_idx,current_tokens):

                        span = txt[start_idx:end_idx]

                        if '\n' in span.strip():
                            indices = '{} '.format(start_idx)
                            lf_index = [i + start_idx for i, x in enumerate(span.strip()) if x == '\n']
                            for l in lf_index:
                                indices += '{};{} '.format(l, (l + 2))
                            indices += '{}'.format(end_idx)
                            wf.write('T{}\t{} {}\t{}\n'.format(Tindex, current_tokens[0]['label'], indices,
                                                                span.replace('\n', '')))

                        else:
                            wf.write(
                                'T{}\t{} {} {}\t{}\n'.format(Tindex, label, start_idx, end_idx,
                                                             span))

                        if current_tokens[0]['categorical_value']!='':
                            wf.write('A{}\t{}Value T{} {}\n'.format(Aindex, label, Tindex, current_tokens[0]['categorical_value']))
                            Aindex += 1
                        else:
                            wf.write('E{}\t{}:T{}\n'.format(Eindex, label, Tindex))
                            Eindex += 1
                        current_tokens = []
                        Tindex += 1

                        return Tindex,Eindex,Aindex,current_tokens

                    if tag=='S':
                        if current_tokens.__len__() > 0:
                            Tindex, Eindex, Aindex, current_tokens = writeToken(Tindex, Eindex, Aindex,  current_tokens[0]['label'], current_tokens[0]['index'], current_tokens[0]['index'] + len( current_tokens[0]['token']), current_tokens)

                        Tindex, Eindex, Aindex, current_tokens = writeToken(Tindex, Eindex, Aindex, label, tokens_indices[i], tokens_indices[i]+len(token),   current_tokens)

                    elif tag in ['E'] and current_tokens.__len__()>0  \
                        and current_tokens[-1]['label'] == label and current_tokens[-1]['tag'] in ['B', 'I'] \
                            and current_tokens[-1]['categorical_value'] == categorical_value:
                        Tindex, Eindex, Aindex, current_tokens = writeToken(Tindex, Eindex, Aindex,
                                                                            current_tokens[0]['label'],
                                                                            current_tokens[0]['index'],
                                                                            tokens_indices[i] + len(token), current_tokens)

                    elif tag =='I'  :
                        if current_tokens.__len__()>0 and current_tokens[-1]['label'] == label and current_tokens[-1]['categorical_value'] == categorical_value and current_tokens[-1]['tag'] in ['B', 'I'] :
                            current_tokens.append({'index': tokens_indices[i], 'label': label, 'tag': tag, 'token': token})
                        else:
                            current_tokens = []
                    elif tag == 'B':
                        current_tokens=[{'index': tokens_indices[i], 'label': label, 'tag': tag, 'token': token,'categorical_value':categorical_value}]
                    else:
                        current_tokens = []

                elif schema=='IOB2':

                    def writeToken(Tindex,Eindex,Aindex,current_tokens):
                        start_idx = current_tokens[0]['index']
                        end_idx = current_tokens[-1]['index'] + current_tokens[-1]['token'].__len__()
                        span=txt[start_idx:end_idx]

                        if '\n' in span.strip():
                            indices = '{} '.format(start_idx)
                            lf_index = [i + start_idx for i, x in enumerate(span.strip()) if x == '\n']
                            for l in lf_index:
                                indices += '{};{} '.format(l, (l + 2))
                            indices += '{}'.format(end_idx)
                            wf.write('T{}\t{} {}\t{}\n'.format(Tindex, current_tokens[0]['label'], indices, span.replace('\n', '')))

                        else:
                            wf.write(
                                'T{}\t{} {} {}\t{}\n'.format(Tindex, current_tokens[0]['label'], start_idx,  end_idx,span ))

                        if current_tokens[0]['categorical_value']!='':
                            wf.write('A{}\t{}Value T{} {}\n'.format(Aindex, current_tokens[0]['label'], Tindex, current_tokens[0]['categorical_value']))
                            Aindex += 1
                        else:
                            wf.write(
                                'E{}\t{}:T{}\n'.format(Eindex, current_tokens[0]['label'], Tindex))
                            Eindex += 1

                        current_tokens=[]
                        Tindex += 1
                        return Tindex,Eindex,Aindex,current_tokens

                    if tag in ['O'] and current_tokens.__len__() > 0  and current_tokens[-1]['tag'] in ['B', 'I']:
                        Tindex, Eindex, Aindex, current_tokens=writeToken(Tindex,Eindex,Aindex,current_tokens)

                    elif tag == 'I':
                        if current_tokens.__len__() > 0 and current_tokens[-1]['label'] == label  and current_tokens[-1]['categorical_value'] == categorical_value and current_tokens[-1][
                            'tag'] in ['B', 'I']:
                            current_tokens.append({'index': tokens_indices[i], 'label': label, 'tag': tag, 'token': token, 'categorical_value':categorical_value})
                        elif  current_tokens.__len__() > 0  and not ( current_tokens[-1]['label'] == label  and current_tokens[-1]['categorical_value'] == categorical_value):
                            Tindex, Eindex, Aindex, current_tokens = writeToken(Tindex, Eindex, Aindex, current_tokens)
                        else:
                            current_tokens = []
                    elif tag == 'B':
                        if  current_tokens.__len__() > 0  and current_tokens[-1]['tag'] in ['B', 'I']:
                            Tindex, Eindex, Aindex, current_tokens = writeToken(Tindex, Eindex, Aindex, current_tokens)

                        current_tokens = [{'index': tokens_indices[i], 'label': label, 'tag': tag, 'token': token,'categorical_value':categorical_value}]
                    else:
                        current_tokens = []


    def extract_entities(self,file):
        filepath, ann_file = os.path.split(file)
        ann_file = os.path.join(filepath, ann_file.split('.')[0] + '.ann')
        ann_lines = open(ann_file, 'r').readlines()
        entities = self.read_brat_entities(file, ann_lines)

        entities_to_T = {}
        for en in entities:
            entities_to_T['{}-{}-{}'.format(entities[en]['span'], entities[en]['start'], entities[en]['end'])] \
                = {'Tindex': en, 'label': entities[en]['label']}

        return entities_to_T, entities

    def extract_entities_relations(self,file, events, preds=None):
        entities_to_T, entities = self.extract_entities(file)
        relation_map = {}
        for i, data in enumerate(events):

            from_token = '{}-{}-{}'.format(data['h']['name'], data['h']['start_idx'], data['h']['end_idx'])
            to_token = '{}-{}-{}'.format(data['t']['name'], data['t']['start_idx'], data['t']['end_idx'])

            if preds is not None and preds[i] != 'no_relation':
                if from_token not in relation_map:
                    relation_map[from_token] = []

                relation_map[from_token].append({'t': to_token, 'relation': preds[i]})
            elif data['relation'] is not None:
                if from_token not in relation_map:
                    relation_map[from_token] = []

                relation_map[from_token].append({'t': to_token, 'relation': data['relation']})

        return entities_to_T, entities, relation_map

    def save_brat_re(self,file, out_folder, events, preds  ):
        ann_file = file + '.ann'
        txt_file = file + '.txt'
        entities_to_T, entities, relation_map= self.extract_entities_relations(txt_file, events,preds)

        copyfile(txt_file, os.path.join(out_folder,os.path.basename(txt_file)))
        copyfile(ann_file, os.path.join(out_folder,os.path.basename(ann_file)))

        events_map={}
        ann_lines=[]
        e_indices_map={}
        with open(ann_file, 'r') as rf:
            ann_lines= rf.readlines()
            E_lines = [l for l in ann_lines if l.startswith('E')]
            for l in E_lines:
                events_map[l.split()[1].split(':')[1]]=l

        relation_end_Ts=set()
        for i, rels in enumerate(relation_map):
            if entities_to_T[rels]['Tindex'] in events_map:
                e_line =events_map[entities_to_T[rels]['Tindex']].rstrip()
                e_idx=e_line.split()[0]
                for r in relation_map[rels]:
                    e_line+= ' {}:{}'.format(r['relation'], entities_to_T[r['t']]['Tindex'])
                    relation_end_Ts.add(entities_to_T[r['t']]['Tindex'])
                e_line+='\n'
                events_map[entities_to_T[rels]['Tindex']] = e_line
                e_indices_map[e_idx]=e_line

        elines = []
        with open(os.path.join(out_folder, os.path.basename(ann_file)), 'w') as wf:
            for l in ann_lines:
                l_tokens = l.split()
                if (len(l_tokens) > 0):
                    if l_tokens[0] not in e_indices_map and not l.startswith('E'):
                        wf.write(l)
                    elif  l.startswith('E') and  l_tokens[0] not in e_indices_map and l_tokens.__len__()==2 and l_tokens[1].split(':')[1] not in relation_end_Ts:
                        wf.write(l)
                    elif  l_tokens[0] in e_indices_map :
                        elines .append(e_indices_map[l_tokens[0]])

            for e in elines:
                wf.write(e)


    def read_brat_events(self,brat_folder, unlabelled=False,distance=None, entities_labels=None, relation_labels=None,   no_relation_ratio=None,max_sent_windows=None,num_workers=1):
        # Logger.debug('reading brat files in folder ', brat_folder)

        ann_files, txt_files= self.get_brat_files(brat_folder)

        sortedFiles = [join(brat_folder, splitext(a)[0]) for a in ann_files]
        sortedFiles.sort()

        event_files = {}
        distances = []
        for f, fname in enumerate(sortedFiles):
            event_files.update(self.brat_to_events(fname,distances=distances, entities_labels=entities_labels, relation_labels=relation_labels))

        if len(distances):
            Logger.info('distance (no. of chars) between entities in relations - min: {} mean: {} median: {} max: {}'.format(
                int(np.min(distances)), int(np.mean(distances)), int(np.median(distances)), int(np.max(distances))
            ))
        elif not unlabelled:
            Logger.error('training data contains no events.')

        if not unlabelled and distance is None:
            distance = int(np.median(distances))

        return self.events_to_dict(event_files,distance=distance, unlabelled=unlabelled,
                              no_relation_ratio=no_relation_ratio,max_sent_windows=max_sent_windows, num_workers=num_workers)

    def read_brat_entities(self,txt_file, ann_lines, labels=None):
        txt = open(txt_file,newline='',mode='rU').read()
        sents = [s for s in nlp(txt).doc.sents]
        sents_start_idx = []
        idx=0
        for s in sents:
            sents_start_idx.append(idx)
            idx+=len(s.text_with_ws)

        def find_sent_idx(start_index):
            for i, s in enumerate(sents_start_idx):
                if start_index<s:
                    return i-1
            return i

        entities = {}
        for ann_line in ann_lines:
            anns = ann_line.split()
            if (ann_line).startswith('T'):
                entity_id = anns[0]
                label = anns[1]
                if labels != None and label not in [l.strip() for l in labels]: continue

                start_index,end_index=self.read_T_indices(ann_line)
                span = txt[start_index:end_index]
                entities[entity_id] = {'label': label, 'start': start_index, 'end': end_index, 'span': span, 'sent_idx':find_sent_idx(start_index)}

        return entities

    def brat_to_events(self,fname,distances = [], entities_labels=None,relation_labels=None ):

        event_files={}

        with open(fname + '.ann', 'r') as ann_file:

            events = {}
            ann_lines = ann_file.readlines()
            entities = self.read_brat_entities(fname+'.txt', ann_lines,entities_labels)
            for ann_line in ann_lines:
                anns = ann_line.split()
                if (ann_line).startswith('E'):
                    event_id = anns[0]

                    start_entity_label,start_entity_id = anns[1].split(':')
                    if(start_entity_id not in entities.keys()): continue
                    events[event_id] = []
                    for event in anns[2:]:
                        event_label, end_entity_id = event.split(':')
                        if (end_entity_id not in entities.keys()) or (relation_labels is not None and event_label not in relation_labels) :
                            continue
                        elif relation_labels is None:
                            while event_label[-1].isnumeric():
                                event_label = event_label[0:-1]

                        events[event_id].append({'label':event_label, 'start_entity_id': start_entity_id, 'end_entity_id': end_entity_id })
                        distances.append(entities[end_entity_id]['start']-entities[start_entity_id]['end']
                                         if entities[start_entity_id]['start']<entities[end_entity_id]['start'] else
                                         entities[start_entity_id]['start'] - entities[end_entity_id]['end'])


            event_files[fname]={'events':events, 'entities':entities}


        return event_files


    def events_to_dict(self,event_files, event_entities=None, distance=None, unlabelled=False,  no_relation_ratio=None,max_sent_windows=None,  num_workers=1):
        events_dict=[]
        for i,file in enumerate(tqdm(list(event_files.keys())) if not unlabelled else list(event_files.keys())):
            events = event_files[file]['events']
            entities= event_files[file]['entities']
            file_events_dict=[]
            no_relation_events = []
            txt_file = file+ '.txt'
            text_gold = open(txt_file, newline='', mode='rU').read()

            if len(events)>0 and not unlabelled:
                for eid in events.keys():
                    e_list=events[eid]
                    for e in e_list:
                        start_entity_start_idx = entities[e['start_entity_id']]['start']
                        start_entity_end_idx = entities[e['start_entity_id']]['end']
                        end_entity_start_idx = entities[e['end_entity_id']]['start']
                        end_entity_end_idx = entities[e['end_entity_id']]['end']
                        start_span = entities[e['start_entity_id']]['span']
                        end_span = entities[e['end_entity_id']]['span']
                        start_span_label = entities[e['start_entity_id']]['label']
                        end_span_label = entities[e['end_entity_id']]['label']
                        label = e['label']

                        if len(set(range(start_entity_start_idx, start_entity_end_idx)).intersection(range(end_entity_start_idx, end_entity_end_idx))) > 0:
                            continue

                        text, start_entity_start_idx, start_entity_end_idx, end_entity_start_idx,end_entity_end_idx = \
                            self.handle_span_with_subtokens(text_gold, start_span, end_span, start_entity_start_idx, start_entity_end_idx, end_entity_start_idx,end_entity_end_idx,start_span_label,end_span_label)

                        sents = list([s.text_with_ws for s in nlp(text).sents])

                        event_dict = self.to_jsonl_dict(txt_file, text, sents, label, start_span, end_span,
                                                         start_entity_start_idx, start_entity_end_idx, end_entity_start_idx,end_entity_end_idx,start_span_label, end_span_label )
                        if event_dict.__len__()>0:
                            file_events_dict.append(event_dict)
                no_relation_events.extend(self.add_no_relation(txt_file,file_events_dict,distance=distance,max_sent_windows=max_sent_windows,no_relation_ratio=no_relation_ratio, num_workers=num_workers))
                events_dict.extend(file_events_dict)
                events_dict.extend(no_relation_events)

            elif unlabelled:
                entities_to_T, entities = self.extract_entities(file+ '.txt')
                events_dict = self.create_relations(file + '.txt', label=None,entities=entities, event_entities=event_entities, labelled_events=None, distance=distance,
                                               max_sent_windows=max_sent_windows,
                                               max_relations=None,  num_workers=num_workers)

        for i, e in enumerate(events_dict):
            e['sample_idx'] = i

        return events_dict

    def to_jsonl_dict(self,file, text, sents, label, start_span, end_span ,  start_span_start_idx, start_span_end_idx, end_span_start_idx,end_span_end_idx ,start_span_label, end_span_label ):

        output_sents=[]
        idx=0
        sent_start_idx=-1

        end_span_before_start_span = (end_span_start_idx<start_span_start_idx and end_span_end_idx <=start_span_start_idx)

        for i, s in enumerate(sents):
            if idx+len(s)>start_span_start_idx and not end_span_before_start_span:
                output_sents.append(s)
                if sent_start_idx==-1:
                    sent_start_idx=idx
            elif idx+len(s)>end_span_start_idx and end_span_before_start_span:
                output_sents.append(s)
                if sent_start_idx == -1:
                    sent_start_idx = idx
            idx+=len(s)
            assert ''.join([s for s in sents[0:i+1]])==text[0:idx]

            if (idx > end_span_end_idx and not end_span_before_start_span) or (
                idx > start_span_end_idx and end_span_before_start_span):
                break
        output_sents_txt = ''.join(output_sents)

        tokens = self.tokenize(output_sents_txt)

        start_tokens = []
        for tok in self.tokenize(text[sent_start_idx:start_span_start_idx]):
            start_tokens.append( self.tokenize(tok) if self._use_spacy else [tok] )
        start_pos = sum([len(v) for v in start_tokens])

        end_tokens = []
        for tok in self.tokenize(text[sent_start_idx:end_span_start_idx]):
            end_tokens.append( self.tokenize(tok) if self._use_spacy else [tok] )
        end_pos = sum([len(v) for v in end_tokens ])

        start_span_tokenized = self.tokenize(start_span)
        true_start_span = ' '.join(start_span_tokenized).strip(punctuation)
        parsed_start_span = ' '.join(tokens[start_pos : start_pos+start_span_tokenized.__len__()]).strip(punctuation)
        if not true_start_span in parsed_start_span  :

            Logger.debug('parsed token "{}" not the same as expected token "{} in {}".  Ignoring .... '
                        .format(parsed_start_span,  true_start_span ,file))
            return {}

        assert true_start_span in parsed_start_span
        end_span_tokenized = self.tokenize(end_span)
        true_end_span = ' '.join(end_span_tokenized)
        parsed_end_span = ' '.join(tokens[end_pos: end_pos + end_span_tokenized .__len__()])
        if(not parsed_end_span.startswith(true_end_span)):

            Logger.debug(file, label, start_span, end_span, start_span_start_idx, start_span_end_idx, end_span_start_idx,
                    end_span_end_idx,start_span_label, end_span_label)
            Logger.debug('parsed token "{}" not the same as expected token "{} in {}".  Ignoring .... '
                         .format(parsed_end_span, true_end_span, file))

            return {}


        assert parsed_end_span.startswith(true_end_span)


        return {'filename': file, 'token': tokens,
                'h': {"name": start_span, "pos": [start_pos , start_pos+ start_span_tokenized.__len__()], 'start_idx':start_span_start_idx, 'end_idx': start_span_end_idx, 'label': start_span_label} ,
                't': {"name": end_span, "pos": [end_pos, end_pos + end_span_tokenized .__len__()], 'start_idx':end_span_start_idx, 'end_idx': end_span_end_idx, 'label': end_span_label
                                             }, "relation": label}

    def create_relations(self,txt_file, label, entities, event_entities=None, labelled_events= None, distance=None, max_sent_windows=None, max_relations=None,  num_workers=1):
        text_gold = open(txt_file, newline='', mode='rU').read()
        sents = list([s.text_with_ws for s in nlp(text_gold).sents])
        events = []
        done=False
        entities = {k: v for k, v in sorted(entities.items(), key=lambda item: item[1]['start'])}
        for en1 in entities:
            for en2 in entities:
                if en1 != en2:
                    start_entity_start_idx = entities[en1]['start']
                    start_entity_end_idx = entities[en1]['end']
                    end_entity_start_idx = entities[en2]['start']
                    end_entity_end_idx = entities[en2]['end']
                    start_span = entities[en1]['span']
                    end_span = entities[en2]['span']
                    start_span_label = entities[en1]['label']
                    end_span_label = entities[en2]['label']
                    start_entity_sent_idx = entities[en1]['sent_idx']
                    end_entity_sent_idx = entities[en2]['sent_idx']

                    labelled_en1 = '{}-{}-{}'.format(entities[en1]['span'], entities[en1]['start'], entities[en1]['end'])
                    labelled_en2 = '{}-{}-{}'.format(entities[en2]['span'], entities[en2]['start'],   entities[en2]['end'])
                    if label == 'no_relation' and labelled_en1 in labelled_events and any([l['t'] == labelled_en2 for l in labelled_events[labelled_en1]]):
                        continue

                    if pd.Interval(start_entity_start_idx,start_entity_end_idx).overlaps(pd.Interval(end_entity_start_idx,end_entity_end_idx)) :
                        continue

                    if max_sent_windows is not None and abs(start_entity_sent_idx-end_entity_sent_idx)>=max_sent_windows:
                        if end_entity_sent_idx>start_entity_sent_idx:
                            break
                        else:
                            continue

                    if max_sent_windows is None and distance is not None:
                        if start_entity_start_idx < end_entity_start_idx:
                            if end_entity_start_idx - start_entity_end_idx > distance:
                                continue
                        else:
                            if start_entity_start_idx - end_entity_end_idx > distance:
                                continue

                    if labelled_events is not None:
                        start_entity_key='{}-{}-{}'.format(start_span, start_entity_start_idx, start_entity_end_idx)
                        end_entity_key = '{}-{}-{}'.format(end_span, end_entity_start_idx, end_entity_end_idx)
                        if start_entity_key in labelled_events and end_entity_key in [e['t'] for e in labelled_events[start_entity_key]]:
                            continue

                    if event_entities is not None and start_span_label+'|'+end_span_label not in event_entities:
                        continue

                    text=text_gold
                    if labelled_events is not None:
                        text, start_entity_start_idx, start_entity_end_idx, end_entity_start_idx,end_entity_end_idx = \
                            self.handle_span_with_subtokens(text_gold, start_span, end_span, start_entity_start_idx, start_entity_end_idx, end_entity_start_idx,end_entity_end_idx,start_span_label,end_span_label )
                        sents = list([s.text_with_ws for s in nlp(text).sents])

                    events.append((txt_file, text, sents, label, start_span, end_span,
                                   start_entity_start_idx, start_entity_end_idx,
                                   end_entity_start_idx, end_entity_end_idx,start_span_label,end_span_label  ))

        if max_relations is not None and max_relations<len(events):
            events = random.sample(events, max_relations)

        with Pool(processes=(num_workers if num_workers > 1 else (None if num_workers == 0 else 1))) as pool:
            events_dict = pool.starmap(self.to_jsonl_dict, iter(events))

        return [ e for e in events_dict if len(e) > 0 ]

    def add_no_relation(self,txt_file, events, distance=None, max_sent_windows=None, no_relation_ratio=None,   num_workers=1):

        max_relations = None
        if no_relation_ratio is not None:
            max_relations = int(len(events) * no_relation_ratio)

        entities_to_T, entities, labelled_events = self.extract_entities_relations(txt_file, events)
        relations= self.create_relations(txt_file,'no_relation', entities=entities, event_entities=None, labelled_events=labelled_events, distance=distance,
                                    max_sent_windows=max_sent_windows, max_relations=max_relations,    num_workers=num_workers)
        return relations


def rad_report_format(lines):

    formated=''
    for line in lines:
        if (line.strip().__len__()>0):
            lastchar=line.strip()[-1]
            if lastchar.isalnum():
                formated+=line.strip()+' '
            else:
                formated+=line
    return formated

