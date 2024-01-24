from transformers.models.layoutlmv3.tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from transformers.models.layoutlmv3.tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast
from transformers.utils import logging


logger = logging.get_logger(__name__)

class GraphLayoutLMTokenizer(LayoutLMv3Tokenizer):
    pass

class GraphLayoutLMTokenizerFast(LayoutLMv3TokenizerFast):
    pass
