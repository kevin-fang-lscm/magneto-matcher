from .utils import get_samples, detect_column_type

modes = [
    "header_values_default",
    "header_values_prefix",
    "header_values_repeat",
    "header_values_verbose",
    "header_only",
    "header_values_simple"
]

sampling_modes = ["random", "frequent", "mixed"]

class ColumnEncoder:
    def __init__(self, tokenizer, encoding_mode="header_values_repeat", sampling_mode="mixed", n_samples=10):

        self._tokenizer = tokenizer
        self._serialization_methods = {
            "header_values_default": self._serialize_header_values_default,
            "header_values_prefix": self._serialize_header_values_prefix,
            "header_values_repeat": self._serialize_header_values_repeat,
            "header_values_verbose": self._serialize_header_values_verbose,
            "header_only": self._serialize_header_only,
            "header_values_simple": self._serialize_header_values_simple
        }
        
        if encoding_mode not in self._serialization_methods:
            raise ValueError(f"Unsupported mode: {encoding_mode}. Supported modes are: {list(self._serialization_methods.keys())}")
        if sampling_mode not in sampling_modes:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}. Supported modes are: {sampling_modes}")
            
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.n_samples = n_samples

    def encode(self, df, col):

        header = col
        tokens = get_samples(df[col], n=self.n_samples, mode=self.sampling_mode)
        data_type = detect_column_type(df[col])
        return self._serialization_methods[self.encoding_mode](header, data_type, tokens)

    def _serialize_header_values_verbose(self, header, data_type, tokens):
        return (
            self._tokenizer.cls_token
            + "Column: " + header
            + self._tokenizer.sep_token
            + "Type: " + data_type
            + self._tokenizer.sep_token
            + "Values: " + self._tokenizer.sep_token.join(tokens)
            + self._tokenizer.sep_token
            + self._tokenizer.eos_token
        )

    def _serialize_header_values_default(self, header, data_type, tokens):
        return f"{self._tokenizer.cls_token}{header}{self._tokenizer.sep_token}{data_type}{self._tokenizer.sep_token}{self._tokenizer.sep_token.join(tokens)}"

    def _serialize_header_values_prefix(self, header, data_type, tokens):
        return f"{self._tokenizer.cls_token}header:{header}{self._tokenizer.sep_token}datatype:{data_type}{self._tokenizer.sep_token}values:{', '.join(tokens)}"

    def _serialize_header_values_repeat(self, header, data_type, tokens):
        return f"{self._tokenizer.cls_token}{self._tokenizer.sep_token.join([header] * 5)}{self._tokenizer.sep_token}{data_type}{self._tokenizer.sep_token}{self._tokenizer.sep_token.join(tokens)}"

    def _serialize_header_only(self, header, data_type, tokens):

        return f"{self._tokenizer.cls_token}{header}{self._tokenizer.eos_token}"

    def _serialize_header_values_simple(self, header, data_type, tokens):

        return (
            self._tokenizer.cls_token
            + "Column: " + header
            + self._tokenizer.sep_token
            + "Values: " + self._tokenizer.sep_token.join(tokens)
            + self._tokenizer.sep_token
            + self._tokenizer.eos_token
        )