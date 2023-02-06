from ts.torch_handler.base_handler import BaseHandler
#import sentencepiece as spm
import torch
import os
import logging
from tempfile import NamedTemporaryFile

import sacremoses
from subword_nmt.apply_bpe import BPE
from onmt.online_translator import TranslatorParameter, OnlineTranslator
from onmt.inference.fast_translator import FastTranslator

logger = logging.getLogger(__name__)


class LanguageTranslationHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        self._context = context
        self.initialized = True
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu")

        bpe_codes = os.path.join(model_dir, 'codes')
        with open(bpe_codes, encoding='utf-8') as codes:
            self.bpe = BPE(codes)

        self.truecaser = sacremoses.MosesTruecaser(f"{model_dir}/Smartcasemodel")
        self.detruecaser = sacremoses.MosesDetruecaser()
        self.tokenizer = sacremoses.MosesTokenizer()
        self.detokenizer = sacremoses.MosesDetokenizer()

        #filename = os.path.join(model_dir, 'model.conf')
        #self.translator = OnlineTranslator(filename)
        with NamedTemporaryFile('w') as f:
            f.write(f'''
model {model_dir}/model.pt
beam_size 4
no_repeat_ngram_size 4
            '''.strip())
            f.flush()
            #self.translator = OnlineTranslator(f.name)
            self.model_opt = TranslatorParameter(f.name)
        self.translator = FastTranslator(self.model_opt)

        self.initialized = True

    def preprocess(self, data):
        def _process(sentence):
            textpp = " ".join(self.tokenizer.tokenize(sentence))
            textpp = " ".join(self.truecaser.truecase(textpp))
            textpp = self.bpe.process_line(textpp)
            return textpp

        inputs = []
        prefixes = []
        for row in data:
            text = row.get("data") or row.get("body") or row.get("text")
            decoded_text = text.decode('utf-8')
            val = _process(decoded_text)
            inputs.append(val)

            prefix = row.get("prefix")
            if prefix is not None:
                prefixes.append(_process(prefix.decode('utf-8')))
            else:
                prefixes.append("")

        if not any(prefixes):
            prefixes = None

        return inputs, prefixes

    def inference(self, data):
        sentences, prefixes = data
        print(sentences)
        print(prefixes)
        #translations = [self.translator.translate(s) for s in sentences]
        inp = [s.split() for s in sentences]
        out = self.translator.translate(inp, [], prefix=prefixes)
        translations = [" ".join(out_i[0]) for out_i in out[0]]
        return translations

    def postprocess(self, data):
        # for sentencepiece detokenization
        #return [x.replace(" ", "").replace("\u2581", " ").strip() for x in data]
        def _process(sentence):
            pp = sentence.replace('@@ ', '')
            pp = self.detokenizer.detokenize(pp.split(" "))
            #pp = pp.capitalize()
            pp = ' '.join(self.detruecaser.detruecase(pp))
            return pp
        return [_process(s) for s in data]
