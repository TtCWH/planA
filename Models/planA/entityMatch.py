import tensorflow as tf
import sys
import os

sys.path.append("../")
from Models import config_utils
from metrics import *
from preprocess import mmd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# hyper parameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability(Default: 0.5)") # dropput比例
tf.flags.DEFINE_integer("train_epoch", 2, "number of epochs to run when training(Default: 5)") # 训练轮数
tf.flags.DEFINE_integer("embedding_dim", 100, "word embedding dimension(Default: 100)") # 词向量维度
tf.flags.DEFINE_integer("batch_size", 128, "batch_size(Defualt: 128)") # batch大小
tf.flags.DEFINE_bool("check_mode", False, "check_mode(Defailt:False)") # 
tf.flags.DEFINE_string("task", "cal", "task(Default:train)") # 训练还是测试
tf.flags.DEFINE_string("model_name","_2","model name") # 

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


class EntityMatch(object):
    def __init__(self, config, relation_size, entity_size):
        self.config = config

        self.max_adj_num = int(config.get("data_config", "max_adj_entities")) # 一度连接最大实体数
        self.embed_dim = FLAGS.embedding_dim 
        self.relation_size = relation_size # 总关系数

        self.pos_h = tf.placeholder(tf.int32, [None, relation_size, self.max_adj_num], name="pos_h") # 正样本 head 一度关系实体 最多10
        self.pos_t = tf.placeholder(tf.int32, [None, relation_size, self.max_adj_num], name="pos_t") # 正样本 tail 一度关系实体 最多10
        self.pos_h_len = tf.placeholder(tf.int32, [None, relation_size], name="pos_h_len") # 正样本 head 一度关系实体个数
        self.pos_t_len = tf.placeholder(tf.int32, [None, relation_size], name="pos_t_len") # 正样本 tail 一度关系实体个数

        self.neg_h = tf.placeholder(tf.int32, [None, relation_size, self.max_adj_num], name="neg_h")
        self.neg_t = tf.placeholder(tf.int32, [None, relation_size, self.max_adj_num], name="neg_t")
        self.neg_h_len = tf.placeholder(tf.int32, [None, relation_size], name="neg_h_len")
        self.neg_t_len = tf.placeholder(tf.int32, [None, relation_size], name="neg_t_len")

        self.y = tf.placeholder(tf.int32, [None], name="y") 

        self.r = tf.placeholder(tf.int32, [], name="r") # 用哪一个卷积核filter

        self.entity_embedding = tf.get_variable("entity_embedding", [entity_size, self.embed_dim], tf.float32,
                                                tf.truncated_normal_initializer(0, 0.2))
        self.relation_filters = tf.get_variable("filters_w", [relation_size, 3, 3, 1, 8], tf.float32,
                                                tf.truncated_normal_initializer(0, 0.2)) # 每个relation对应一个卷积核[3,3,1,8]
        self.relation_b = tf.get_variable("filters_b", [relation_size, 8], tf.float32, tf.zeros_initializer())

        self.mask_pad_embeeding = tf.concat([tf.zeros([1, 1]), tf.ones([self.entity_embedding.get_shape()[0] - 1, 1])], axis=0)
         # 把第一个embedding全置为0
        # self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)

        self.drop_keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout")
        self.batch_size = tf.shape(self.pos_h)[0]

        self.get_emb() # 获取embedding
        self.match() # 
        # self.match_logloss()

    def get_emb(self):
        self.pos_h_emb = tf.nn.embedding_lookup(self.entity_embedding, self.pos_h)
        mask_pad = tf.nn.embedding_lookup(self.mask_pad_embeeding, self.pos_h)
        self.pos_h_emb = tf.multiply(self.pos_h_emb, mask_pad) # 把pad的embedding全置为0

        self.pos_t_emb = tf.nn.embedding_lookup(self.entity_embedding, self.pos_t)
        mask_pad = tf.nn.embedding_lookup(self.mask_pad_embeeding, self.pos_t)
        self.pos_t_emb = tf.multiply(self.pos_t_emb, mask_pad)

        self.neg_h_emb = tf.nn.embedding_lookup(self.entity_embedding, self.neg_h)
        mask_pad = tf.nn.embedding_lookup(self.mask_pad_embeeding, self.neg_h)
        self.neg_h_emb = tf.multiply(self.neg_h_emb, mask_pad)

        self.neg_t_emb = tf.nn.embedding_lookup(self.entity_embedding, self.neg_t)
        mask_pad = tf.nn.embedding_lookup(self.mask_pad_embeeding, self.neg_t)
        self.neg_t_emb = tf.multiply(self.neg_t_emb, mask_pad)

        self.pos_h_emb = tf.reduce_sum(self.pos_h_emb, axis=2) 
        self.pos_h_emb = self.pos_h_emb / tf.to_float(tf.expand_dims(self.pos_h_len, axis=2))
         # 对每个关系所取的max_adj_num个实体的embedding取均值

        self.pos_t_emb = tf.reduce_sum(self.pos_t_emb, axis=2)
        self.pos_t_emb = self.pos_t_emb / tf.to_float(tf.expand_dims(self.pos_t_len, axis=2))

        self.neg_h_emb = tf.reduce_sum(self.neg_h_emb, axis=2)
        self.neg_h_emb = self.neg_h_emb / tf.to_float(tf.expand_dims(self.neg_h_len, axis=2))

        self.neg_t_emb = tf.reduce_sum(self.neg_t_emb, axis=2)
        self.neg_t_emb = self.neg_t_emb / tf.to_float(tf.expand_dims(self.neg_t_len, axis=2))

    def match(self):
        self.filter_w = tf.nn.embedding_lookup(self.relation_filters, self.r) # 选择对应的filter
        self.filter_b = tf.nn.embedding_lookup(self.relation_b, self.r)

        # pos_pool_dim = pos_pool.get_shape().as_list()
        # flatten_dim = pos_pool_dim[1] * pos_pool_dim[2] * pos_pool_dim[3]
        # print "flatten_dim: ", flatten_dim
        flatten_dim = 2888
        w1 = tf.get_variable("w1", [flatten_dim, 10], tf.float32, tf.truncated_normal_initializer(0, 0.01))
        b1 = tf.get_variable("b1", [10], tf.float32, tf.constant_initializer())

        w2 = tf.get_variable("w2", [10, 1], tf.float32, tf.truncated_normal_initializer(0, 0.01))
        b2 = tf.get_variable("b2", [1], tf.float32, tf.constant_initializer())

        # pos ######
        pos_cross = tf.einsum("abd,acd->abc", self.pos_h_emb, self.pos_t_emb)
         # [batchsize,relation_size,embedding] * [batchsize,relation_size,embedding] -> [batchsize,relation_size,relation_size] 形成一个相似矩阵
        # mask_l = tf.sign(tf.reshape(self.pos_h_len, [self.batch_size, self.relation_size, 1]))
        # mask_r = tf.sign(tf.reshape(self.pos_t_len, [self.batch_size, 1, self.relation_size]))
        # mask = mask_l + mask_r
        # mask = tf.cast(tf.greater(mask, 0), dtype=tf.float32)
        #  '''只保留两个关系都有连接实体的部分，即都为1的部分'''
        # pos_cross = tf.multiply(pos_cross, mask)
        # self.pos_cross = pos_cross
        # pos_cross = tf.nn.dropout(pos_cross, self.drop_keep_prob)

        pos_cross_img = tf.expand_dims(pos_cross, axis=3)
        pos_conv = tf.nn.relu(tf.nn.conv2d(pos_cross_img, self.filter_w, [1, 1, 1, 1], "SAME", name="pos_conv") + self.filter_b,
                              name="relu_pos")
        pos_pool = tf.nn.max_pool(pos_conv, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        pos_pool_flatten = tf.reshape(pos_pool, [self.batch_size, flatten_dim])
        # pos_pool_flatten = tf.nn.dropout(pos_pool_flatten, self.drop_keep_prob)
        pos_fc1 = tf.nn.relu(tf.matmul(pos_pool_flatten, w1) + b1)
        pos_fc1 = tf.nn.dropout(pos_fc1, self.drop_keep_prob)
        pos_score = tf.nn.sigmoid(tf.matmul(pos_fc1, w2) + b2)
        self.pos_logits = pos_score

        # neg #####
        neg_cross = tf.einsum("abd,acd->abc", self.neg_h_emb, self.neg_t_emb)
        # mask_l = tf.sign(tf.reshape(self.neg_h_len, [self.batch_size, self.relation_size, 1]))
        # mask_r = tf.sign(tf.reshape(self.neg_t_len, [self.batch_size, 1, self.relation_size]))
        # mask = mask_l + mask_r
        # mask = tf.cast(tf.greater(mask, 0), dtype=tf.float32)
        # neg_cross = tf.multiply(neg_cross, mask)
        # self.neg_cross = neg_cross
        # neg_cross = tf.nn.dropout(neg_cross, self.drop_keep_prob)

        neg_cross_img = tf.expand_dims(neg_cross, axis=3)
        neg_conv = tf.nn.relu(tf.nn.conv2d(neg_cross_img, self.filter_w, [1, 1, 1, 1], "SAME", name="neg_conv") + self.filter_b,
                              name="relu_neg")
        neg_pool = tf.nn.max_pool(neg_conv, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        neg_pool_flatten = tf.reshape(neg_pool, [self.batch_size, flatten_dim])
        # neg_pool_flatten = tf.nn.dropout(neg_pool_flatten, self.drop_keep_prob)
        neg_fc1 = tf.nn.relu(tf.matmul(neg_pool_flatten, w1) + b1)
        neg_fc1 = tf.nn.dropout(neg_fc1, self.drop_keep_prob)
        neg_score = tf.nn.sigmoid(tf.matmul(neg_fc1, w2) + b2)
        self.neg_logits = neg_score

        # loss #####
        self.l2_loss = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
        self.loss = tf.reduce_mean(tf.maximum(1.0 + neg_score - pos_score, 0)) + 0.001 * self.l2_loss

    def match_logloss(self):
        self.filter_w = tf.nn.embedding_lookup(self.relation_filters, self.r)
        self.filter_b = tf.nn.embedding_lookup(self.relation_b, self.r)

        flatten_dim = 2888
        w1 = tf.get_variable("w1", [flatten_dim, 10], tf.float32, tf.truncated_normal_initializer(0, 0.01))
        b1 = tf.get_variable("b1", [10], tf.float32, tf.constant_initializer())

        w2 = tf.get_variable("w2", [10, 2], tf.float32, tf.truncated_normal_initializer(0, 0.01))
        b2 = tf.get_variable("b2", [2], tf.float32, tf.constant_initializer())

        pos_cross = tf.einsum("abd,acd->abc", self.pos_h_emb, self.pos_t_emb)
        mask_l = tf.sign(tf.reshape(self.pos_h_len, [self.batch_size, self.relation_size, 1]))
        mask_r = tf.sign(tf.reshape(self.pos_t_len, [self.batch_size, 1, self.relation_size]))
        mask = mask_l + mask_r
        mask = tf.cast(tf.greater(mask, 0), dtype=tf.float32)
        pos_cross = tf.multiply(pos_cross, mask)

        pos_cross_img = tf.expand_dims(pos_cross, axis=3)
        pos_conv = tf.nn.relu(tf.nn.conv2d(pos_cross_img, self.filter_w, [1, 1, 1, 1], "SAME", name="pos_conv") + self.filter_b,
                              name="relu_pos")
        pos_pool = tf.nn.max_pool(pos_conv, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        pos_pool_flatten = tf.reshape(pos_pool, [self.batch_size, flatten_dim])
        pos_fc1 = tf.nn.relu(tf.matmul(pos_pool_flatten, w1) + b1)
        pos_fc1 = tf.nn.dropout(pos_fc1, self.drop_keep_prob)
        pos_score = tf.matmul(pos_fc1, w2) + b2
        self.pos_logits = tf.nn.softmax(pos_score)

        # loss #####
        self.l2_loss = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pos_score, labels=self.y)) + 0.001 * self.l2_loss

class EntityMatchTaskRunner(object):
    @staticmethod
    def train():
        config = config_utils.get_config() # 获取超参
        # model_data = MatchModelData(config)
        model_data = mmd
        train_iter = model_data.train_batch_iter(FLAGS.train_epoch, FLAGS.batch_size) # 获取batch训练数据

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True, # 智能分配gpu
                log_device_placement=False) # 不打印设备log
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.3 #每个GPU拿出30%的容量给进程使用
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = EntityMatch(config, len(model_data.relations) + 1, len(model_data.entities) + 1)
                 # 模型，关系数为relation*2+1(其本身)，实体数为实体数+1(pad的embedding)
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(0.001)
                grads_and_vars = optimizer.compute_gradients(model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                def train_step():
                    batch_pos_h, batch_pos_t, batch_neg_h, batch_neg_t, batch_pos_h_len, batch_pos_t_len, batch_neg_h_len, \
                    batch_neg_t_len, r = train_iter.next()
                    # batch_pos_h, batch_pos_t, batch_pos_h_len, batch_pos_t_len, y, r = train_iter.next()
                    feed_dict = {
                        model.pos_h: batch_pos_h,
                        model.pos_t: batch_pos_t,
                        model.pos_h_len: batch_pos_h_len,
                        model.pos_t_len: batch_pos_t_len,
                        model.neg_h: batch_neg_h,
                        model.neg_t: batch_neg_t,
                        model.neg_h_len: batch_neg_h_len,
                        model.neg_t_len: batch_neg_t_len,
                        # model.y: y,
                        model.r: r,
                        model.drop_keep_prob: FLAGS.dropout_keep_prob
                    }
                    # _, loss = sess.run([train_op, model.loss], feed_dict)
                    # return loss, r

                    _, loss, pos = sess.run([train_op, model.loss, model.pos_logits], feed_dict)
                    return loss, r, pos

                    # _, loss, pos, neg = sess.run([train_op, model.loss, model.pos_logits, model.neg_logits], feed_dict)
                    # return loss, r, pos, neg

                    # _, loss, pos_c, neg_c = sess.run([train_op, model.loss, model.pos_cross, model.neg_cross], feed_dict)
                    # return loss, r, pos_c, neg_c

                    # _, loss, pos_c, neg_c = sess.run([train_op, model.loss, model.pos_h_emb, model.pos_h_len], feed_dict)
                    # return loss, r, pos_c, neg_c

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                model_path = model.config.get("model_path", "entityMatch")
                p = model_path.rfind("/") # 找到model_path最后一次出现 '/' 的位置
                model_path = model_path[:p] + FLAGS.model_name + model_path[p:]

                tf.add_to_collection("logits_op", model.pos_logits)
                # tf.add_to_collection("neg_logits", model.neg_logits)

                loss, relation_id = 0, 0
                pos, neg = 0, 0
                while True:
                    try:
                        for _ in range(100):
                            # loss, relation_id = train_step()
                            loss, relation_id, pos = train_step()
                            # loss, relation_id, pos, neg = train_step()
                            if (_+1)%10==0:
                                print("save model")
                                saver.save(sess, model_path)
                        print "train loss {}, relation_id {}".format(loss, relation_id)
                        print "pos: ", pos[0]
                        # print "neg: ", neg[0]
                    except StopIteration:
                        print "training end."
                        saver.save(sess, model_path)
                        break

    @staticmethod
    def evaluate():
        config = config_utils.get_config() # 获取配置
        # model_data = MatchModelData(config)
        model_data = mmd

        eval_iter = model_data.eval_batch_iter(batch_size=1000) # 获取测试集
        pred_results_path = model_data.config.get("predictions_path", "entityMatch")
        if mmd.only_test_part:
            pred_results_path = pred_results_path.replace(".csv", mmd.file_postfix)

        model_path = model_data.config.get("model_path", "entityMatch")
        p = model_path.rfind("/")
        model_path = model_path[:p] + FLAGS.model_name + model_path[p:]

        graph = tf.get_default_graph()
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.25
        with tf.Session(graph=graph,config=session_conf) as sess:
            saver = tf.train.import_meta_graph(model_path + ".meta")
            saver.restore(sess, model_path)

            logits_op = tf.get_collection("logits_op")[0]

            pos_h_ph = graph.get_tensor_by_name("pos_h:0")
            pos_h_len_ph = graph.get_tensor_by_name("pos_h_len:0")
            pos_t_ph = graph.get_tensor_by_name("pos_t:0")
            pos_t_len_ph = graph.get_tensor_by_name("pos_t_len:0")

            neg_h_ph = graph.get_tensor_by_name("neg_h:0")
            neg_h_len_ph = graph.get_tensor_by_name("neg_h_len:0")
            neg_t_ph = graph.get_tensor_by_name("neg_t:0")
            neg_t_len_ph = graph.get_tensor_by_name("neg_t_len:0")

            drop_ph = graph.get_tensor_by_name("dropout:0")

            r_ph = graph.get_tensor_by_name("r:0")

            # neg_logits = tf.get_collection("neg_logits")[0]

            print "model restored."

            def eval_step():
                batch_h, batch_h_len, batch_t, batch_t_len, h, t, r, e_ids = eval_iter.next()
                feed_dict = {
                    pos_h_ph: batch_h,
                    pos_t_ph: batch_t,
                    pos_h_len_ph: batch_h_len,
                    pos_t_len_ph: batch_t_len,
                    # neg_h_ph: batch_h,
                    # neg_t_ph: batch_t,
                    # neg_h_len_ph: batch_h_len,
                    # neg_t_len_ph: batch_t_len,
                    r_ph: r,
                    drop_ph: 1.0
                }
                y_pred = sess.run([logits_op], feed_dict)
                # y_pred = sess.run([neg_logits], feed_dict)
                return h, t, r, e_ids, y_pred

            f = open(pred_results_path, "w")
            cnt = 0
            while True:
                try:
                    h, t, r, e_ids, y_pred = eval_step()
                    # print y_pred
                    probs = map(lambda x: x[0], y_pred[0])
                    # probs = map(lambda x: x[0], y_pred)
                    probs = ",".join(map(str, probs))
                    new_line = ",".join(map(str, [h, t, r])) + "|" + probs
                    f.write(new_line + "\n")

                    cnt += 1
                    if not cnt % 100:
                        print "has predicted {} batchs".format(cnt)
                except StopIteration:
                    f.close()
                    print "evaluation done."
                    break


if __name__ == "__main__":
    task = FLAGS.task
    if task == "train":
        EntityMatchTaskRunner.train()
    elif task == "eval":
        EntityMatchTaskRunner.evaluate()
    elif task == "cal":
        config = config_utils.get_config()
        process_predict_result(config, config.get("predictions_path", "entityMatch"), first_n=10)
