# Magneto

Magneto is an innovative framework designed to enhance schema matching (SM) by intelligently combining small, pre-trained language models (SLMs) with large language models (LLMs). Our approach is structured to be both cost-effective and broadly applicable.

## Installation


You can install the latest stable version of Magneto from [PyPI](https://pypi.org/project/magneto-python/):

```
pip install magneto-python
```


## Usage
After the installation, you can use the stand-alone version of Magneto like this:

```Python
from magneto import Magneto
import pandas as pd

source = pd.DataFrame({"column_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]})
target = pd.DataFrame({"column_1a": ["a1", "b1", "c1"], "col2": ["a2", "b2", "c2"]})

mode = "header_values_verbose"
mag = Magneto(encoding_mode=mode)
matches = mag.get_matches(source, target)

print(matches)
```

See our [GitHub repository](https://github.com/VIDA-NYU/data-integration-eval/tree/main/algorithms/magneto) for more examples.