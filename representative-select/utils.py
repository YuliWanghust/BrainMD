import torch
import openai
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import GPTJForCausalLM
from collections import OrderedDict
import sqlparse

def calculate_sentence_transformer_embedding(text_to_encode,args):
    num = len(text_to_encode)
    emb_model = SentenceTransformer(args.embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    return embeddings

def codex_execution(key,output_path,prompt_path):
    openai.api_key = key
    with open(prompt_path) as f:
        prompt = json.load(f)[1]
    completion = openai.Completion.create(
        engine='code-davinci-002',
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        logprobs=5,
        stop=['--', '\n\n', ';', '#'],
    )
    with open(output_path, 'w') as f:
        json.dump(completion, f)

def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]

PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))
def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())
    return set(answers)

def slot_values_to_seq_sql(original_slot_values, single_answer=False):
    sql_str = ""
    tables = OrderedDict()
    col_value = dict()

    # add '_' in SQL columns
    slot_values = {}
    for slot, value in original_slot_values.items():
        if ' ' in slot:
            slot = slot.replace(' ', '_')
        slot_values[slot] = value

    for slot, value in slot_values.items():
        assert len(slot.split("-")) == 2

        if '|' in value:
            value = value.split('|')[0]

        table, col = slot.split("-")  # slot -> table-col

        if table not in tables.keys():
            tables[table] = []
        tables[table].append(col)

        # sometimes the answer is ambiguous
        if single_answer:
            value = value.split('|')[0]
        col_value[slot] = value

    # When there is only one table
    if len(tables.keys()) == 1:
        where_clause = []
        table = list(tables.keys())[0]
        for col in tables[table]:
            where_clause.append("{} = {}".format(col, col_value["{}-{}".format(table, col)]))
        sql_str = "SELECT * FROM {} WHERE {}".format(table, " AND ".join(where_clause))
    # When there are more than one table
    else:
        # We observed that Codex has variety in the table short names, here we just use a simple version.
        from_clause = []
        where_clause = []
        for i, table in enumerate(tables.keys()):
            t_i = "t{}".format(i + 1)
            from_clause.append("{} AS {}".format(table, t_i))
            for col in tables[table]:
                where_clause.append("{}.{} = {}".format(t_i, col, col_value["{}-{}".format(table, col)]))
        sql_str = "SELECT * FROM {} WHERE {}".format(", ".join(from_clause), " AND ".join(where_clause))

    return sql_str

class PreviousStateRecorder:

    def __init__(self):
        self.states = {}

    def add_state(self, data_item, slot_values):
        dialog_ID = data_item['dialogue_ID']
        turn_id = data_item['turn_id']
        if dialog_ID not in self.states:
            self.states[dialog_ID] = {}
        self.states[dialog_ID][turn_id] = slot_values

    def state_retrieval(self, data_item):
        dialog_ID = data_item['dialogue_ID']
        turn_id = data_item['turn_id']
        if turn_id == 0:
            return {}
        else:
            return self.states[dialog_ID][turn_id - 1]

def codex_completion(prompt_text,key,output_path,model_name='code-davinci-002'):
    openai.api_key = key
    result = openai.Completion.create(
        engine=model_name,
        prompt=prompt_text,
        max_tokens=200,
        temperature=0,
        logprobs=5,
        stop=['--', '\n', ';', '#'],
    )
    with open(output_path, 'w') as f:
        json.dump(result, f)
    return result["choices"][0]["text"]

def sql_pred_parse(pred):
    # parse sql results and fix general errors

    pred = " * FROM" + pred

    # fix for no states
    if pred == " * FROM  WHERE ":
        return {}

    # Here we need to write a parser to convert back to dialogue state
    pred_slot_values = []
    # pred = pred.lower()
    parsed = sqlparse.parse(pred)
    if not parsed:
        return {}
    stmt = parsed[0]
    sql_toks = pred.split()
    operators = [" = ", " LIKE ", " < ", " > ", " >= ", " <= "]

    if "AS" in pred:
        as_indices = [i for i, x in enumerate(sql_toks) if x == "AS"]

        table_name_map_dict = {}
        for indice in as_indices:
            table_name_map_dict[sql_toks[indice + 1].replace(",", "")] = sql_toks[indice - 1]

        slot_values_str = str(stmt.tokens[-1]).replace("_", " ").replace("""'""", "").replace("WHERE ", "")
        for operator in operators:
            slot_values_str = slot_values_str.replace(operator, "-")
        slot_values = slot_values_str.split(" AND ")

        for sv in slot_values:
            for t_ in table_name_map_dict.keys():
                sv = sv.replace(t_ + ".", table_name_map_dict[t_] + "-")
            pred_slot_values.append(sv)
    else:

        table_name = sql_toks[sql_toks.index("FROM") + 1]

        slot_values_str = str(stmt.tokens[-1]).replace("_", " ").replace("""'""", "").replace("WHERE ", "")
        for operator in operators:
            slot_values_str = slot_values_str.replace(operator, "-")
        slot_values = slot_values_str.split(" AND ")

        pred_slot_values.extend([table_name + "-" + sv for sv in slot_values if slot_values != ['']])

    pred_slot_values = {'-'.join(sv_pair.split('-')[:-1]): sv_pair.split('-')[-1] for sv_pair in pred_slot_values}

    # remove _ in SQL columns
    pred_slot_values = {slot.replace('_', ' '): value for slot, value in pred_slot_values.items()}

    # fix typos
    # pred_slot_values, _ = typo_fix(pred_slot_values)

    return pred_slot_values

def check_prefix_suffix(value, candidates):
    # add/delete "the" in the front, or the suffix in the end.
    if value in candidates:
        return value
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    for prefix in [''] + prefixes:
        for suffix in [''] + suffixes:
            possible_value = prefix + value + suffix
            if possible_value in candidates:
                return possible_value
    return ''

def compute_acc(gold, pred, n_slot=30):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = n_slot
    ACC = n_slot - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def evaluate(preds: dict, golds: dict):

    gold_slots = list(golds.keys())
    for k in gold_slots:
        if '|' in golds[k]:
            gold_values = golds[k].split('|')
            if k in preds and preds[k] in gold_values:
                golds[k] = preds[k]

    jga, acc, f1 = 0, 0, 0

    if preds == golds:
        jga = 1
    acc = compute_acc(golds, preds)
    f1 = compute_prf(golds, preds)[0]

    return jga, acc, f1
