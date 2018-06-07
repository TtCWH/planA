# coding:utf-8
from numpy.random import binomial
from collections import defaultdict, OrderedDict
import sys
import numpy as np
from random import choice, shuffle
from multiprocessing import Process, Queue, Pool, Manager

sys.path.append("../")
from Models import config_utils


class MatchModelData(object):
    def __init__(self, config):
        self.config = config

        # 一些配置
        self.max_adj_entities = int(config.get("data_config", "max_adj_entities"))
        self.max_relation_num = int(config.get("data_config", "max_relation_num"))

        self.entity_graph = {}
        self.entity_graph_id = {}
        self.entity_graph_len = {}
        self.relation2id = OrderedDict()
        self.inverse_r = {}  # 记录每个关系的inverse 关系 (都是id)
        self.entity2id = OrderedDict()
        self.pos_triples_dict = {}
        self.pos_triples_per_relation = {}

        self.test_triples = []
        self.related_relations = {}  # 对于每个关系,取与它最相关的max_relation_num个关系
        self.r_row_id = {}           # 对于每个关系而言,其它关系的相邻样本在其临近矩阵中的哪一行
        self.total_row_id = {}       # 记录在总的entity_graph中,每个关系在临近矩阵中的哪一行

        self.read_relation_dict()
        self.read_entity_dict()
        self.generate_graph_one_hop()
        self.generate_graph_id_and_len()
        self.split_data_by_relation()
        self.read_related_relations()
        print "generate mmd done."

        self.entities = self.entity2id.keys()
        self.entities_id = self.entity2id.values()
        self.relations = self.relation2id.keys()
        self.relations_id = self.relation2id.values()

        self.n_process = 3
        self.max_que_size = 20
        self.que = Queue(self.max_que_size)
        self.eval_que = Manager().Queue(self.max_que_size)

        self.only_train_one_relation = False
        self.specific_relation = 14

        self.only_test_part = True
        # self.part_test_relation = range(0, 5)
        # self.file_postfix = "_1.csv"

        # self.part_test_relation = range(5, 10)
        # self.file_postfix = "_2.csv"

        self.part_test_relation = range(10, 18)
        self.file_postfix = "_3.csv"

    def read_relation_dict(self):
        relation_dict_path = self.config.get("model_data_path", "wn_relation2id")
        f = open(relation_dict_path)
        for line in f:
            relation, relation_id = line.strip().split("\t")
            self.relation2id[relation] = int(relation_id)

        relation_id = len(self.relation2id)
        keys = self.relation2id.keys()
        for r in keys:
            reverse_r = r + '_r'
            self.relation2id[reverse_r] = relation_id
            self.inverse_r[relation_id] = self.relation2id[r]
            self.inverse_r[self.relation2id[r]] = relation_id
            relation_id += 1

        # 记录每个关系的相邻实体在每个实体的邻接实体矩阵的哪一行, key是每个关系的id
        r_id = 1  # r_id = 0 is for (h,t) itself !!!
        for r in self.relation2id:
            self.total_row_id[self.relation2id[r]] = r_id
            r_id += 1
        print "read relation dict done."

    def read_entity_dict(self):
        entity_dict_path = self.config.get("model_data_path", "wn_word2id")
        f = open(entity_dict_path)
        for line in f:
            entity, entity_id = line.strip().split()
            self.entity2id[entity] = int(entity_id) + 1  # 0 is for padding
        print "read entity dict done."

    def generate_graph_one_hop(self):
        train_data_path = self.config.get("model_data_path", "wn_train")
        f = open(train_data_path)
        for line in f:
            head, tail, relation = line.strip().split("\t")

            if head not in self.entity_graph:
                self.entity_graph[head] = {}
                for r in self.relation2id:
                    self.entity_graph[head][r] = []
            if len(self.entity_graph[head][relation]) < self.max_adj_entities:
                self.entity_graph[head][relation].append(tail)

            if tail not in self.entity_graph:
                self.entity_graph[tail] = {}
                for r in self.relation2id:
                    self.entity_graph[tail][r] = []
            if len(self.entity_graph[tail][relation + '_r']) < self.max_adj_entities:
                self.entity_graph[tail][relation + '_r'].append(head)

            pos_triple = (self.entity2id[head], self.entity2id[tail], self.relation2id[relation])
            self.pos_triples_dict[pos_triple] = 1
        print "generate graph one hop done."

    def read_related_relations(self):
        f = open(self.config.get("model_data_path","wn_related"))
        for line in f:
            r_id, other = line.strip().split(": ")
            r_id = int(r_id)
            self.related_relations[r_id] = {}
            self.r_row_id[r_id] = OrderedDict()

            terms = other.split()
            len_terms = len(terms)
            for i in range(self.max_relation_num):
                if i < len_terms:
                    t_id = int(terms[i].split(",")[0][1:])
                    self.r_row_id[r_id][t_id] = i + 1 #0 is for (h,t) itself!!!
                else:
                    pass

        print "read related relations done."

    def write_related_relations(self):
        train_data_path = self.config.get("model_data_path", "wn_train")
        f = open(train_data_path)
        for line in f:
            head, tail, relation = line.strip().split("\t")

            if relation not in self.related_relations:
                self.related_relations[relation] = {}

            for r in self.relation2id:
                if len(self.entity_graph[head][r]):
                    if r not in self.related_relations[relation]:
                        self.related_relations[relation][r] = 0
                    self.related_relations[relation][r] += len(self.entity_graph[head][r]) # 1
                if len(self.entity_graph[tail][r]):
                    if r not in self.related_relations[relation]:
                        self.related_relations[relation][r] = 0
                    self.related_relations[relation][r] += len(self.entity_graph[tail][r])

        g = open(self.config.get("model_data_path","wn_related"), "w")
        for r in self.related_relations:
            new_line = self.relation2id[r] + ": "
            for r1 in self.related_relations[r]:
                new_line += "(" + self.relation2id[r1] + "," + str(self.related_relations[r][r1]) + ") "
            g.write(new_line + "\n")

    def read_test_triples(self):
        test_data_path = self.config.get("model_data_path", "wn_test")
        f = open(test_data_path)
        for line in f:
            head, tail, relation = line.strip().split("\t")
            head_id, tail_id, relation_id = self.entity2id[head], self.entity2id[tail], self.relation2id[relation]
            self.test_triples.append((head_id, tail_id, relation_id))
        print "read test triples done."

    def generate_graph_id_and_len(self):
        for e in self.entity_graph:
            e_id = self.entity2id[e]
            if e_id not in self.entity_graph_id:
                self.entity_graph_id[e_id] = [[e_id]]
            if e_id not in self.entity_graph_len:
                self.entity_graph_len[e_id] = [1]
            for r in self.relation2id:
                self.entity_graph_id[e_id].append(self._map_entity_to_id(self.entity_graph[e][r]))
                l = len(self.entity_graph[e][r])
                if l == 0:
                    self.entity_graph_len[e_id].append(-1)
                else:
                    self.entity_graph_len[e_id].append(l)
        print "generate graph id and len done."

    def split_data_by_relation(self):
        for h, t, r in self.pos_triples_dict:
            if r not in self.pos_triples_per_relation:
                self.pos_triples_per_relation[r] = []
            self.pos_triples_per_relation[r].append((h, t, r))

    def _map_entity_to_id(self, entity_list):
        a = map(lambda x: self.entity2id[x], entity_list)
        return a

    def _padding(self, entity_adj):
        size = len(entity_adj)
        for i in range(size):
            entity_adj[i].extend([0] * (self.max_adj_entities - len(entity_adj[i])))

    def write_adj_data(self, datatype):
        f = open(self.config.get("model_data_path", "wn_{}".format(datatype)))
        g = open(self.config.get("model_data_path", "wnm_{}".format(datatype)), "w")
        for line in f:
            g.write(line)

            head, tail, relation = line.strip().split("\t")
            for r in self.relation2id:
                head_adj_entities = self.entity_graph[head][r]
                tail_adj_entities = self.entity_graph[tail][r]
                new_line = "  |  ".join([r, ",".join(head_adj_entities), ",".join(tail_adj_entities)])
                g.write(new_line + "\n")
            g.write("\n")
        print "generate adjacent entities for {} data done.".format(datatype)

    def sample_negative_triple(self, pos_triple):
        to_replace_head = binomial(1, 0.5)
        while True:
            random_e = choice(self.entities_id)
            if to_replace_head:
                corrupted_triple = (random_e, pos_triple[1], pos_triple[2])
                if corrupted_triple not in self.pos_triples_dict:
                    break
            if not to_replace_head:
                corrupted_triple = (pos_triple[0], random_e, pos_triple[2])
                if corrupted_triple not in self.pos_triples_dict:
                    break
        return corrupted_triple

    def count_batch_num(self, n_epoch, batch_size):
        batch_num = 0
        for r in self.pos_triples_per_relation:
            if self.only_train_one_relation and r != self.specific_relation:  # remove this!!!!!!!
                continue

            sample_num = len(self.pos_triples_per_relation[r])
            batch_num += int((sample_num - 1) / batch_size) + 1
            print "r: ", r, " sample_num: ", sample_num, " batch_num: ", int((sample_num - 1) / batch_size) + 1
        batch_num *= n_epoch
        print "total batch_num of {} epochs: ".format(n_epoch, batch_size), batch_num
        return batch_num

    def generate_one_sample_train_data(self, triple):
        h, t, r = triple
        inverse_r = self.inverse_r[r]

        h_adj, h_adj_len, t_adj, t_adj_len = [[h]], [1], [[t]] ,[1]
        for r_ in self.r_row_id[r]:     #从每个实体的全部临近矩阵中取出需要的那些行
            h_adj.append(self.entity_graph_id[h][self.total_row_id[r_]])
            h_adj_len.append(self.entity_graph_len[h][self.total_row_id[r_]])
            t_adj.append(self.entity_graph_id[t][self.total_row_id[r_]])
            t_adj_len.append(self.entity_graph_len[t][self.total_row_id[r_]])

            if r_ == r and t in h_adj[-1]: #如果当前r和当前triple的r相同,去掉出现在r中的实体h,r,以免标注结果出现在特征中
                h_adj[-1].remove(t)
                h_adj_len[-1] -= 1
                if h_adj_len[-1] == 0:
                    h_adj_len[-1] = -1
            if r_ == inverse_r and h in t_adj[-1]:
                t_adj[-1].remove(h)
                t_adj_len[-1] -= 1
                if t_adj_len[-1] == 0:
                    t_adj_len[-1] = -1

        self._padding(h_adj)
        self._padding(t_adj)
        return h_adj, h_adj_len, t_adj, t_adj_len

    def _train_data_producer(self, n_epoch, batch_size):
        for i in range(n_epoch):
            for r in self.pos_triples_per_relation:
                if self.only_train_one_relation and r != self.specific_relation:  # remove this!!!!!!!!!
                    continue

                sample_num = len(self.pos_triples_per_relation[r])
                num_batches_per_epoch = int((sample_num - 1) / batch_size) + 1

                shuffle(self.pos_triples_per_relation[r])
                for j in range(num_batches_per_epoch):
                    start_index = j * batch_size
                    end_index = min((j + 1) * batch_size, sample_num)

                    batch_pos_triple = self.pos_triples_per_relation[r][start_index:end_index]
                    batch_neg_triple = []
                    for t in batch_pos_triple:
                        batch_neg_triple.append(self.sample_negative_triple(t))

                    batch_pos_h, batch_pos_t, batch_neg_h, batch_neg_t, batch_pos_h_len, batch_pos_t_len, batch_neg_h_len, \
                    batch_neg_t_len = [], [], [], [], [], [], [], []

                    for t in batch_pos_triple:
                        h, h_len, t, t_len = self.generate_one_sample_train_data(t)
                        batch_pos_h.append(h)
                        batch_pos_h_len.append(h_len)
                        batch_pos_t.append(t)
                        batch_pos_t_len.append(t_len)

                    for t in batch_neg_triple:
                        h, h_len, t, t_len = self.generate_one_sample_train_data(t)
                        batch_neg_h.append(h)
                        batch_neg_h_len.append(h_len)
                        batch_neg_t.append(t)
                        batch_neg_t_len.append(t_len)

                    batch_pos_h = np.array(batch_pos_h)
                    batch_pos_t = np.array(batch_pos_t)
                    batch_pos_h_len = np.array(batch_pos_h_len)
                    batch_pos_t_len = np.array(batch_pos_t_len)

                    batch_neg_h = np.array(batch_neg_h)
                    batch_neg_t = np.array(batch_neg_t)
                    batch_neg_h_len = np.array(batch_neg_h_len)
                    batch_neg_t_len = np.array(batch_neg_t_len)

                    self.que.put([batch_pos_h, batch_pos_t, batch_neg_h, batch_neg_t, batch_pos_h_len, batch_pos_t_len,
                                  batch_neg_h_len, batch_neg_t_len, r])

    def train_batch_iter(self, n_epoch, batch_size):
        epoch_per_process = n_epoch / self.n_process
        for i in range(self.n_process):
            producer_process = Process(target=self._train_data_producer, args=(epoch_per_process, batch_size))
            producer_process.start()

        total_batch_num = self.count_batch_num(epoch_per_process, batch_size) * self.n_process
        for i in range(total_batch_num):
            data = self.que.get()
            yield data

    def count_per_relation_test_sample_num(self):
        ans = defaultdict(int)
        for h_, t_, r_ in self.test_triples:
            ans[r_] += 1
        return ans

    def eval_batch_iter(self, batch_size):
        self.read_test_triples()
        per_relation_num = self.count_per_relation_test_sample_num()

        pool = Pool(processes=self.n_process)

        entities_num = len(self.entities)
        batch_num = int((entities_num - 1) / batch_size) + 1
        for h_, t_, r_ in self.test_triples:
            if self.only_train_one_relation and r_ != self.specific_relation:  # remove this
                continue
            if self.only_test_part and r_ not in self.part_test_relation:
                continue

            pool.apply_async(eval_batch_generate, args=(h_,t_,r_,batch_size, batch_num, entities_num))

        if self.only_train_one_relation:
            total_batch_num = per_relation_num[self.specific_relation] * batch_num
            print "test sample num {}, total batch num {}".format(per_relation_num[self.specific_relation], total_batch_num)
        elif self.only_test_part:
            triple_num = 0
            for r_id in self.part_test_relation:
                triple_num += per_relation_num[r_id]
            total_batch_num = triple_num * batch_num
            print "test sample num {}, total batch num {}".format(triple_num, total_batch_num)
        else:
            total_batch_num = len(self.test_triples) * batch_num
            print "test sample num {}, total batch num {}".format(len(self.test_triples), total_batch_num)

        for i in range(total_batch_num):
            data = self.eval_que.get()
            yield data

def eval_batch_generate(h_,t_,r_,batch_size, batch_num, entities_num):
    global mmd
    for j in range(batch_num):
        start_index = j * batch_size
        end_index = min((j + 1) * batch_size, entities_num)
        candidate_entity_id = mmd.entities_id[start_index:end_index]

        batch_h, batch_h_len, batch_t, batch_t_len, e_ids = [], [], [], [], []
        for e_id in candidate_entity_id:
            h, h_len, t, t_len = mmd.generate_one_sample_train_data((h_, e_id, r_))
            batch_h.append(h)
            batch_h_len.append(h_len)
            batch_t.append(t)
            batch_t_len.append(t_len)
            e_ids.append(e_id)

        batch_h = np.array(batch_h)
        batch_h_len = np.array(batch_h_len)
        batch_t = np.array(batch_t)
        batch_t_len = np.array(batch_t_len)

        mmd.eval_que.put([batch_h, batch_h_len, batch_t, batch_t_len, h_, t_, r_, e_ids])

config = config_utils.get_config()
mmd = MatchModelData(config)

if __name__ == "__main__":
    # print "done"
    # mmd = MatchModelData(config)
    # h = mmd.entity2id["03964744"]
    # t = mmd.entity2id["04371774"]
    # r = mmd.relation2id["_hyponym"]
    # h = 6068
    # t = 31616
    # r = 5
    # print mmd.entity_graph_id[h]
    # print mmd.entity_graph_id[t]
    # a = mmd.generate_one_sample_train_data((h,t,r))
    # print a[0]
    # print a[1]
    # print a[2]
    # print a[3]

    # for t in mmd.pos_triples_dict:
    #     mmd.generate_one_sample_train_data(t)

    # train_iter = mmd.train_batch_iter(4, 1)
    #
    # cnt = 0
    # for t in train_iter:
    #     cnt += 1
    #     if  cnt > 50:
    #         break
    #     for i in range(9):
    #         print t[i]
    #     print "==================="

    # eval_iter = mmd.eval_batch_iter(1000)
    # cnt = 0
    # for t in eval_iter:
    #     cnt += 1
    #     print cnt
    #
    #     if not cnt % 10:
    #         time.sleep(3)

    MatchModelData.count_batch_num(mmd, 1, 128)
    # MatchModelData.write_related_relations(mmd)

    # for _ in range(10):
    #     a = mmd.sample_negative_triple((952,38769,8))
    #     print a

    # mmd.read_test_triples()

    # f = open("/home/wl/AliScene/model_data/wordNet/related_relations.csv")
    # g = open("/home/wl/AliScene/model_data/wordNet/related_relations_new.csv", "w")
    # for line in f:
    #     r, other = line.strip().split(": ")
    #     new_line = str(mmd.relation2id[r]) + ": "
    #
    #     t = []
    #     related_relations = other.split()
    #     for term in related_relations:
    #         r_, cnt = term[1:-1].split(",")
    #         r_id = mmd.relation2id[r_]
    #         t.append((r_id, int(cnt)))
    #
    #     t.sort(key=lambda x:x[1], reverse=True)
    #     for term in t:
    #         new_line += "(" + str(term[0]) + "," + str(term[1]) + ") "
    #     g.write(new_line + "\n")