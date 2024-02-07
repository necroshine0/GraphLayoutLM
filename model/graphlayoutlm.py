import math
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)

from model.basemodel import LayoutLMv3Model  # from unilm -- ???
from model.layoutlmv3 import LayoutLM3Model
from model.configuration_graphlayoutlm import GraphLayoutLMConfig


class GraphLayoutLMPreTrainedModel(PreTrainedModel):
    config_class = GraphLayoutLMConfig
    base_model_prefix = "graphlayoutlm"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        super(GraphAttentionLayer, self).__init__()
        self.num_attention_heads = int(config.num_attention_heads / 2)
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.final = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        seq_inputs,
        graph_mask,
    ):
        print("GAT Layer seq_inputs shape:".upper(), seq_inputs.shape)
        mixed_query_layer = self.query(seq_inputs)
        key_layer = self.transpose_for_scores(self.key(seq_inputs))
        value_layer = self.transpose_for_scores(self.value(seq_inputs))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        print("GAT Layer key_layer shape:".upper(), key_layer.shape)
        print("GAT Layer value_layer shape:".upper(), value_layer.shape)
        print("GAT Layer query_layer shape:".upper(), query_layer.shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + graph_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1,1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = self.final(context_layer)
        print("GAT outputs shape:".upper(), outputs.shape)
        return outputs


class SubLayerConnection(nn.Module):
    def __init__(self, config):
        super(SubLayerConnection,self).__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-05)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.size = config.hidden_size

    def forward(self, x, graph_mask, sublayer):
        return x + self.dropout(sublayer(self.norm(x), graph_mask))
    

class GraphLayoutLM(GraphLayoutLMPreTrainedModel):
    def __init__(self, config, detection=False, out_features=None, image_only=False):
        super().__init__(config, detection, out_features, image_only)
        self.model_base = LayoutLM3Model(config, detection, out_features, image_only)
        self.graph_attention_layer = GraphAttentionLayer(config)
        self.sublayer = SubLayerConnection(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        valid_span=None,
        graph_mask=None,
    ):
        print("graphlayoutlm input_ids shape:".upper(), input_ids.shape)
        outputs = self.model_base(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
        )

        # see LayoutLMv3Model's .forward()
        if self.model_base.detection:
            return outputs

        print("model_base outputs[0] shape:".upper(), outputs[0].shape)
        assert outputs[0].shape == torch.Size([2, 709, 768])

        print("graph_mask shape:".upper(), graph_mask.shape)
        sequence_output = self.sublayer(outputs[0], graph_mask, self.graph_attention_layer)

        if not return_dict:
            return (sequence_output, outputs[1])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=outputs.pooler_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class GraphLayoutLMClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size*3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GraphLayoutLMForTokenClassification(GraphLayoutLMPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, detection=False, out_features=None, image_only=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.graphlayoutlm = GraphLayoutLM(config, detection, out_features, image_only)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = GraphLayoutLMClassificationHead(config, pool_feature=False)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        graph_mask=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.graphlayoutlm(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
            graph_mask=graph_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
