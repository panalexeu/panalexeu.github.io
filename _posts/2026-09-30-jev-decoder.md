---
title: turning a system one model (jev) into decoder
author: oleksii horchynskyi
---

I was curious about the jev release and decided to play with it.

jev is known not to be a decoder model but rather some form of encoder. It is advertised as a model that evaluates a state and
returns typed answers and probabilities.

jev supports a `Choice` primitive, which returns probabilities over a specified set of choices.

After finding out about this primitive, I immediately had a bunch of dumb ideas, and I'm experimenting with them in this [repo](https://github.com/panalexeu/jev-decoder).

### experiments

Below are some completed experiments:

#### ascii_char_gen.py

The idea is to use jev as an ASCII character-level decoder model via the `Choice` primitive. ASCII symbols are provided as choices, and the state is concatenated with characters sampled from the returned character probability distribution.

Instruction: `Choose the next ASCII character to generate a coherent continuation of the provided Shakespeare text.`

State: `First Citizen:\nBefore we proceed any further`

Criteria:
```python
def _form_ascii_dict():
    return {chr(i): chr(i) for i in range(128)}
```

Results of generating 16 new symbols:
```text
First Citizen:
Before we proceed any further t h h h   h Ai u
```

#### token_gen.py

The idea is once again to use jev as a decoder model, but this time asking it to generate tokens. The `Choice` primitive accepts a maximum of 255 choices, so a BPE tokenizer with a vocab of 255 is trained on lowercased tiny-shakespeare and provided to `Choice` as criteria. The state is concatenated with tokens sampled from the returned probability distribution.

Instruction: `Choose the next token to generate a coherent continuation of the provided Shakespeare text. Underscore before token means space.`

State: `first citizen:\nbefore we proceed any further`

Criteria:
```python
def _form_token_dict(sp):
    return {sp.id_to_piece(i): sp.id_to_piece(i) for i in range(sp.get_piece_size())}
```

Results of generating 32 tokens:
```text
▁first▁citizen:▁before▁we▁proceed▁any▁further▁but▁king▁let▁letow,▁my▁from▁lord▁to▁he▁me▁king▁he▁let▁myhememe▁king▁and▁'▁'and'▁''▁but?▁king,▁what
```

#### char_eval.py

Evaluate the average loss by asking jev to predict the next character after a 256-character window, and compare it to the loss achieved by uniform random sampling over characters (spoiler: it's much worse than uniform sampling).
Results achieved on 256 consecutive predictions from the start of tiny-shakespeare:

Instruction: `Choose the next ASCII character to generate a coherent continuation of the provided Shakespeare text.`

State:
```python
_idx = -1
_block_size = 256
def _next_sample():
    global _content
    global _idx
    _idx += 1
    return _content[_idx:_block_size+_idx]
```

Criteria:
```python
def _form_shk_vocab():
    global _content
    chars = sorted(list(set(_content)))
    return {ch: ch for ch in chars}
```

Results:
```text
avg. loss: 7.71, uniform sampling: 4.17, catastrophic misses (>20 nats): 45
```

#### sql_gen.py

The idea is again to use jev as a decoder model, but this time the choices come from context-free grammar rules defined for SQL, with one caveat:
NUMBER and STRING tokens are extracted from the correct reference query and added to the choices.
The grammar rules are defined in the `sql.lark` file (I asked Claude to write them). Surprisingly, jev is pretty good at
writing working SQL queries (compared to the previous experiments). On the Spider benchmark, jev's queries return the same results as the reference queries roughly half the time. 489 of 1,034 queries (47.3%) matched the reference results on the Spider dev set.

Instruction: `Choose the next SQL token to generate an SQL query that will answer the question:\n{row['question']}\nSchema:\n{str(schema)}`

State: `''`

Criteria:
```python
def _get_tokens(parser, query: str, ref_query: str, schema: dict) -> set:
    qp = parser.parse_interactive(query)
    qp.exhaust_lexer()
    ref = parser.parse_interactive(ref_query).exhaust_lexer()
    values = {
        "NAME": set(schema) | {c for cs in schema.values() for c in cs} | {f"T{i}" for i in range(1, 6)},
        "AGG_FN": {"COUNT", "SUM", "AVG", "MIN", "MAX"},
        "STRING": {t.value for t in ref if t.type == "STRING"},
        "NUMBER": {t.value for t in ref if t.type == "NUMBER"} or {"1"},
    }
    out = set()
    for t in qp.accepts() - {"$END"}:
        if t in values:
            out |= values[t]
        elif isinstance(p := parser.get_terminal(t).pattern, PatternStr):
            out.add(p.value)
    return out
```

#### sql_gen_cli.py

`sql_gen.py` showed that, given the right context (rules defined by a context-free grammar), jev is somewhat capable of generating valid SQL queries.
`sql_gen_cli.py` lets you prompt jev directly about the `network_1` database from the Spider benchmark.

P.S. Yay, I made jev generate working SQL queries!

Below is the demo:

![demo](/assets/gifs/jev-decoder.gif)