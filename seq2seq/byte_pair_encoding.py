from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["merged.txt"], trainer=trainer)
output = tokenizer.encode("وسنت الهجرة لقادر على إظهاره أي دينه ليتخلص من تكثير الكفار ومخالطتهم ورؤية المنكر بينهم ويتمكن من جهادهم وإعانة المسلمين ويكثرهم")
print(output.tokens)