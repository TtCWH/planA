# coding:utf-8
import numpy as np
from collections import defaultdict
import codecs
from collections import OrderedDict


def round3(x):
    return round(x, 3)


def cal_prob(logits):
    b = np.exp(logits)
    s = np.sum(b, axis=1)
    prob = b[:, 1] / s
    prob = prob.tolist()
    return map(round3, prob)


def cal_avg_precision(pred_list, real_list):
    ap = []
    len_real_list = len(real_list)
    hit_index, cur_index = 0.0, 0
    for p in pred_list:
        cur_index += 1
        if p in real_list:
            hit_index += 1
            ap.append(hit_index / cur_index)
        else:
            ap.append(0)
    return np.sum(ap) / len_real_list


def cal_contains(pred_list, real_list):
    for r in real_list:
        if r in pred_list:
            return 1
    return 0


def hit_n(pred_list, t, n):
    return 1 if t in pred_list[:n] else 0


def mean_rank(pred_list, t):
    return pred_list.index(t) + 1


def mean_rec_rank(pred_list, t):
    return 1.0 / (pred_list.index(t) + 1)


def process_predict_result(config, prediction_path, first_n=3):
    entity2id = OrderedDict()
    id2entity = OrderedDict()
    f = codecs.open(config.get("model_data_path", "wn_word2id"), encoding="utf-8")
    for line in f:
        entity, entity_id = line.strip().split("\t")
        entity2id[entity] = str(int(entity_id) + 1)  # 0 is for padding!!!
        id2entity[entity2id[entity]] = entity

    relation2id = OrderedDict()
    id2relation = OrderedDict()
    f = codecs.open(config.get("model_data_path", "wn_relation2id"), encoding="utf-8")
    for line in f:
        relation, relation_id = line.strip().split("\t")
        relation2id[relation] = relation_id
        id2relation[relation_id] = relation

    test_data_path = config.get("model_data_path", "wn_test")
    test_tail = defaultdict(set)
    f = codecs.open(test_data_path, encoding="utf-8")
    for line in f:
        parts = line.strip().split("\t")
        text1, text2, relation = parts
        h_id, t_id, r_id = entity2id[text1], entity2id[text2], relation2id[relation]
        test_tail[h_id + ":" + r_id].add(t_id)

    train_data_path = config.get("model_data_path", "wn_train")
    train_tail = defaultdict(set)
    f = codecs.open(train_data_path, encoding="utf-8")
    for line in f:
        parts = line.strip().split("\t")
        text1, text2, relation = parts
        h_id, t_id, r_id = entity2id[text1], entity2id[text2], relation2id[relation]
        train_tail[h_id + ":" + r_id].add(t_id)

    index_kinds = 10
    relation_types = {}
    total = [[] for _ in range(index_kinds)]

    ids = map(str, range(1, 1 + len(entity2id)))
    all_predictions = {}
    result_path = prediction_path.replace(".csv", "_pro.csv")
    with codecs.open(prediction_path, encoding="utf-8") as f, codecs.open(result_path, "w", encoding="utf-8") as g:
        for line in f:
            triple, probs = line.strip().split("|")
            h_id, t_id, r_id = triple.split(",")
            probs = map(float, probs.split(","))
            key = ":".join([h_id, t_id, r_id])

            if key not in all_predictions:
                all_predictions[key] = []
            all_predictions[key].extend(probs)

        print "read probs done."

        for k in all_predictions:
            h_id, t_id, r_id = k.split(":")
            probs = all_predictions[k]

            relation = id2relation[r_id]

            if len(probs) < 40943:
                continue
            pairs = zip(ids, probs)
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
            pred_entity = map(lambda x: x[0], pairs)

            if relation not in relation_types:
                relation_types[relation] = [[] for _ in range(index_kinds)]

            mr = mean_rank(pred_entity, t_id)
            mrr = mean_rec_rank(pred_entity, t_id)
            hit_10 = hit_n(pred_entity, t_id, 10)
            hit_3 = hit_n(pred_entity, t_id, 3)
            hit_1 = hit_n(pred_entity, t_id, 1)

            for t_ in test_tail[h_id + ":" + r_id]:
                if t_ != t_id:
                    pred_entity.remove(t_)
            for t_ in train_tail[h_id + ":" + r_id]:
                if t_ != t_id:
                    pred_entity.remove(t_)

            mrf = mean_rank(pred_entity, t_id)
            mrr_f = mean_rec_rank(pred_entity, t_id)
            hit_f_10 = hit_n(pred_entity, t_id, 10)
            hit_f_3 = hit_n(pred_entity, t_id, 3)
            hit_f_1 = hit_n(pred_entity, t_id, 1)

            for i,v in enumerate([mr, mrr, hit_10, hit_3, hit_1, mrf, mrr_f, hit_f_10, hit_f_3, hit_f_1]):
                relation_types[relation][i].append(v)
                total[i].append(v)

            pred_entity = pred_entity[:first_n]
            new_line = ",".join([h_id, r_id, t_id]) + ";  "
            new_line += ",".join([id2entity[h_id], id2relation[r_id], id2entity[t_id]]) + ";  "
            new_line += ",".join(map(lambda x: id2entity[x], pred_entity)) + ";   "
            new_line += ",".join(map(lambda x: id2entity[x], test_tail[h_id + ":" + r_id])) + ";   "
            new_line += ",".join(map(lambda x: id2entity[x], train_tail[h_id + ":" + r_id])) + "\n"
            g.write(new_line)

    for k in relation_types:
        print k.encode('utf-8')
        print "mean rank: ", int(np.mean(relation_types[k][0]))
        print "mean rec rank: ", np.mean(relation_types[k][1])
        print "hit@{}: ".format(10), np.mean(relation_types[k][2])
        print "hit@{}: ".format(3), np.mean(relation_types[k][3])
        print "hit@{}: ".format(1), np.mean(relation_types[k][4])
        print "mean rank filtered: ", int(np.mean(relation_types[k][5]))
        print "mean rec rank filtered: ", np.mean(relation_types[k][6])
        print "hit@{} filtered: ".format(10), np.mean(relation_types[k][7])
        print "hit@{} filtered: ".format(3), np.mean(relation_types[k][8])
        print "hit@{} filtered: ".format(1), np.mean(relation_types[k][9])
        print
    print "in total: "
    print "mean rank: ", int(np.mean(total[0]))
    print "mean rec rank: ", np.mean(total[1])
    print "hit@{}: ".format(10), np.mean(total[2])
    print "hit@{}: ".format(3), np.mean(total[3])
    print "hit@{}: ".format(1), np.mean(total[4])
    print "mean rank filtered: ", int(np.mean(total[5]))
    print "mean rec rank filtered: ", np.mean(total[6])
    print "hit@{} filtered: ".format(10), np.mean(total[7])
    print "hit@{} filtered: ".format(3), np.mean(total[8])
    print "hit@{} filtered: ".format(1), np.mean(total[9])