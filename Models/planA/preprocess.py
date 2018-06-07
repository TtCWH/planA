# coding:utf-8
from numpy.random import binomial
from collections import defaultdict, OrderedDict
import sys
import numpy as np
from random import choice, shuffle
from multiprocessing import Process, Queue, Pool, Manager
from copy import deepcopy

sys.path.append("../")
from Models import config_utils


class MatchModelData(object):
    def __init__(self, config):
        self.config = config

        self.max_adj_entities = int(config.get("data_config", "max_adj_entities"))
        self.entity_graph = {} # key: e_name value: {} -> key[r_name] -> value [adjacent entities names]   36
        self.entity_graph_id = {} # key: e_id value : [[], []] [adjacent entities id]  37
        self.entity_graph_len = {} # key: e_id value: [] v: len([adjacent entities id])  37
        self.relation2id = OrderedDict() # relation:id 
        self.r_row_id = {} # key: r_id value 在entity_graph_id里的哪一行
        self.inverse_r = {} # 记录r的inverse关系
        self.entity2id = OrderedDict() # entity:id
        self.pos_triples_dict = {} #  训练集中所有的triple
        self.pos_triples_per_relation = {} # 训练集中所有的triple按r分类

        self.test_triples = [] # 测试集里的所有triples

        self.read_relation_dict()
        self.read_entity_dict()
        self.generate_graph_one_hop()
        self.generate_graph_id_and_len()
        self.split_data_by_relation()

        self.entities = self.entity2id.keys()
        self.entities_id = self.entity2id.values()
        self.relations = self.relation2id.keys()
        self.relations_id = self.relation2id.values()
        self.max_relation_num = int(config.get("data_config", "max_relation_num"))

        self.negative_times = 1

        self.n_process = 2
        self.max_que_size = 20
        self.que = Queue(self.max_que_size)
        self.eval_que = Manager().Queue(self.max_que_size)

        self.only_train_one_relation = True 
        self.specific_relation = 10

        self.only_test_part = False
        self.part_test_relation = range(0, 5)
        self.file_postfix = "_1.csv"

        # self.part_test_relation = range(5, 10)
        # self.file_postfix = "_2.csv"

        # self.part_test_relation = range(10, 18)
        # self.file_postfix = "_3.csv"

    def read_relation_dict(self):
        '''relation2id key:relation(str) value:id(int)'''
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
            self.r_row_id[self.relation2id[r]] = r_id
            r_id += 1

    def read_entity_dict(self): 
        '''entity2id key:relation(str) value:id(int)'''
        entity_dict_path = self.config.get("model_data_path", "wn_word2id")
        f = open(entity_dict_path)
        for line in f:
            entity, entity_id = line.strip().split()
            self.entity2id[entity] = int(entity_id) + 1  # 0 is for padding

    def generate_graph_one_hop(self):
        '''self.entity_graph 得到实体的一度subgraph
           key:entity_name(str) value:dict
           key:relation_name value:[entity_name(str)]
        '''
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

    def read_test_triples(self):
        '''测试集'''
        test_data_path = self.config.get("model_data_path", "wn_test")
        f = open(test_data_path)
        for line in f:
            head, tail, relation = line.strip().split("\t")
            head_id, tail_id, relation_id = self.entity2id[head], self.entity2id[tail], self.relation2id[relation]
            # print head, tail, relation, head_id, tail_id, relation_id
            self.test_triples.append((head_id, tail_id, relation_id))

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
                    self.entity_graph_len[e_id].append(-1) # 若本实体没有与这个关系相连的实体，则长度为-1
                else:
                    self.entity_graph_len[e_id].append(l)

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

    def sample_negative_triple(self, pos_triple): # 负样本
        """随机替换正样本的头实体 or 尾实体来形成负样本"""
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

    def count_batch_num(self, n_epoch, batch_size): # 计算每个relation训练数据的batch数量
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
        '''在h和t的邻接矩阵里分别去掉t和h'''
        h, t, r = triple
        r_id = self.r_row_id[r]
        inverse_r_id = self.r_row_id[self.inverse_r[r]]

        h_adj = deepcopy(self.entity_graph_id[h])
        h_adj_len = deepcopy(self.entity_graph_len[h])
        t_adj = deepcopy(self.entity_graph_id[t])
        t_adj_len = deepcopy(self.entity_graph_len[t])

        if t in h_adj[r_id]:
            h_adj[r_id].remove(t)
            h_adj_len[r_id] -= 1
            if h_adj_len[r_id] == 0:
                h_adj_len[r_id] = -1
        if h in t_adj[inverse_r_id]:
            t_adj[inverse_r_id].remove(h)
            t_adj_len[inverse_r_id] -= 1
            if t_adj_len[inverse_r_id] == 0:
                t_adj_len[inverse_r_id] = -1

        self._padding(h_adj) # 不够max_adj_entities的，就pad到这么大
        self._padding(t_adj)
        return h_adj, h_adj_len, t_adj, t_adj_len

    def _train_data_producer_logloss(self, n_epoch, batch_size):
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
                        for _ in range(self.negative_times):
                            batch_neg_triple.append(self.sample_negative_triple(t))

                    batch_h, batch_t, batch_h_len, batch_t_len, y = [], [], [], [], []

                    for t in batch_pos_triple:
                        h, h_len, t, t_len = self.generate_one_sample_train_data(t)
                        batch_h.append(h)
                        batch_h_len.append(h_len)
                        batch_t.append(t)
                        batch_t_len.append(t_len)
                        y.append(1)

                    for t in batch_neg_triple:
                        h, h_len, t, t_len = self.generate_one_sample_train_data(t)
                        batch_h.append(h)
                        batch_h_len.append(h_len)
                        batch_t.append(t)
                        batch_t_len.append(t_len)
                        y.append(0)

                    batch_h = np.array(batch_h)
                    batch_t = np.array(batch_t)
                    batch_h_len = np.array(batch_h_len)
                    batch_t_len = np.array(batch_t_len)
                    y = np.array(y)

                    # shuffled_ids = range(len(y))
                    # shuffle(shuffled_ids)
                    # batch_h = batch_h[shuffled_ids]
                    # batch_t = batch_t[shuffled_ids]
                    # batch_h_len = batch_h_len[shuffled_ids]
                    # batch_t_len = batch_t_len[shuffled_ids]
                    # y = y[shuffled_ids]

                    self.que.put([batch_h, batch_t , batch_h_len, batch_t_len, y, r])

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
            # producer_process = Process(target=self._train_data_producer_logloss, args=(epoch_per_process, batch_size))
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
    # WordNetData().combine_all()


    # md = TransEModelData(config)
    # md.read_and_generate_train_test()
    # md.write_statistics_info()

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

    train_iter = mmd.train_batch_iter(2, 1)
    #
    cnt = 0
    for t in train_iter:
        cnt += 1
        print t[4]
        if  cnt > 5:
            break
        # for i in range(9):
        #     print t[i]
        # print "==================="

    # eval_iter = mmd.eval_batch_iter(1000)
    # cnt = 0
    # for t in eval_iter:
    #     cnt += 1
    #     print cnt
    #
    #     if not cnt % 10:
    #         time.sleep(3)

    # MatchModelData.count_batch_num(mmd, 1, 128)

    # for _ in range(10):
    #     a = mmd.sample_negative_triple((952,38769,8))
    #     print a

    # mmd.read_test_triples()