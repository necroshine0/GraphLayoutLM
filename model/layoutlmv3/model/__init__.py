from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from .configuration_layoutlmv3 import LayoutLM3Config
from .modeling_layoutlmv3 import (
    LayoutLM3ForTokenClassification,
    LayoutLM3ForQuestionAnswering,
    LayoutLM3ForSequenceClassification,
    LayoutLM3Model,
)
from .tokenization_layoutlmv3 import LayoutLM3Tokenizer
from .tokenization_layoutlmv3_fast import LayoutLM3TokenizerFast


AutoConfig.register("layoutlm3", LayoutLM3Config)
AutoModel.register(LayoutLM3Config, LayoutLM3Model)
AutoModelForTokenClassification.register(LayoutLM3Config, LayoutLM3ForTokenClassification)
AutoModelForQuestionAnswering.register(LayoutLM3Config, LayoutLM3ForQuestionAnswering)
AutoModelForSequenceClassification.register(LayoutLM3Config, LayoutLM3ForSequenceClassification)
AutoTokenizer.register(
    LayoutLM3Config, slow_tokenizer_class=LayoutLM3Tokenizer, fast_tokenizer_class=LayoutLM3TokenizerFast
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLM3Tokenizer": RobertaConverter})
