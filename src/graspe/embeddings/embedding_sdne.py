# """
# This module originates from library GraphEmbedding (https://github.com/shenweichen/GraphEmbedding)
# """

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2

from embeddings.base.embedding import Embedding


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        b_ = tf.where(
            tf.equal(y_true, 0),
            tf.constant(1, dtype=tf.float32),
            tf.constant(beta, dtype=tf.float32),
        )
        x = K.square((y_true - y_pred) * b_)
        t = K.sum(
            x,
            axis=-1,
        )
        return K.mean(t)

    return loss_2nd


def l_1st(alpha):
    def loss_1st(y_true, y_pred):
        L = y_true
        Y = y_pred
        batch_size = tf.cast(K.shape(L)[0], dtype=tf.float32)
        return (
            alpha
            * 2
            * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y))
            / batch_size
        )

    return loss_1st


def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    A = Input(shape=(node_size,))
    L = Input(shape=(None,))
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(
                hidden_size[i],
                activation="relu",
                kernel_regularizer=l1_l2(l1, l2),
                name="1st",
            )(fc)
        else:
            fc = Dense(
                hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2)
            )(fc)
    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2))(
            fc
        )

    A_ = Dense(node_size, "relu", name="2nd")(fc)
    model = Model(inputs=[A, L], outputs=[A_, Y])
    emb = Model(inputs=A, outputs=Y)
    return model, emb


class SDNEEmbedding(Embedding):
    def __init__(
        self,
        g,
        d,
        hidden_size=[32, 16],
        alpha=1e-6,
        beta=5.0,
        nu1=1e-5,
        nu2=1e-4,
        batch_size=1024,
        epochs=1,
        verbose=1,
    ):
        """
        Parameters
        ----------
        g : common.graph.Graph
            The original graph.
        d : int
            Dimensionality of the embedding.
        epochs : int
            Number of epochs.

        """
        super().__init__(g, d)
        # self.g.remove_edges_from(self.g.selfloop_edges())
        tf.compat.v1.disable_eager_execution()
        self.graph = g.to_networkx()
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)

        self.node_size = self._g.nodes_cnt()
        self.hidden_size = hidden_size
        self.hidden_size.append(d)
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.A, self.L = self._create_A_L(
            self.graph, self.node2idx
        )  # Adj Matrix,L Matrix
        self.inputs = [self.A, self.L]

    def embed(self):
        tf.compat.v1.disable_eager_execution()
        self.model, self.emb_model = create_model(
            self.node_size, hidden_size=self.hidden_size, l1=self.nu1, l2=self.nu2
        )
        self.model.compile("adam", [l_2nd(self.beta), l_1st(self.alpha)])
        self._embeddings = {}
        if self.batch_size >= self.node_size:
            if self.batch_size > self.node_size:
                print(
                    "batch_size({0}) > node_size({1}),set batch_size = {1}".format(
                        self.batch_size, self.node_size
                    )
                )
                self.batch_size = self.node_size
            self.model.fit(
                [self.A.todense(), self.L.todense()],
                [self.A.todense(), self.L.todense()],
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                shuffle=False,
            )
        else:
            steps_per_epoch = (self.node_size - 1) // self.batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, self.epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(
                        i * self.batch_size,
                        min((i + 1) * self.batch_size, self.node_size),
                    )
                    A_train = self.A[index, :].todense()
                    L_mat_train = self.L[index][:, index].todense()
                    inp = [A_train, L_mat_train]
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses = losses / steps_per_epoch

                logs["loss"] = losses[0]
                logs["2nd_loss"] = losses[1]
                logs["1st_loss"] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if self.verbose > 0:
                    print("Epoch {0}/{1}".format(epoch + 1, self.epochs))
                    print(
                        "{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}".format(
                            epoch_time, losses[0], losses[1], losses[2]
                        )
                    )
        self._embedding = {}
        embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embedding[look_back[i]] = embedding

    def evaluate(
        self,
    ):
        return self.model.evaluate(
            x=self.inputs, y=self.inputs, batch_size=self.node_size
        )

    def _create_A_L(self, graph, node2idx):
        node_size = graph.number_of_nodes()
        A_data = []
        A_row_index = []
        A_col_index = []

        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get("weight", 1)

            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])

        A = sp.csr_matrix(
            (A_data, (A_row_index, A_col_index)), shape=(node_size, node_size)
        )
        A_ = sp.csr_matrix(
            (A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
            shape=(node_size, node_size),
        )

        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
        L = D - A_
        return A, L
