from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

class Evaluator():

    def __init__(self, args, dataset, out_dir, test_x, test_chars, task_idx_test, ruling_embedding_test, test_y, batch_size):
        self.dataset = dataset
        self.model_type = args.model_type
        self.out_dir = out_dir
        self.test_x, self.test_chars, self.task_idx_test, self.ruling_embedding_test, self.test_y = test_x, test_chars, task_idx_test, ruling_embedding_test, test_y
        self.test_y = test_y
        self.batch_size = batch_size
        self.best_test_f1 = 0
        self.best_acc = 0
        self.best_report = None
        self.best_test_epoch = -1

    def evaluate(self, model, epoch, print_info=False):
        def get_hateSpeech(test_task_idx, test_y_label, test_pred_label):
            hs_pred, hs_y = [], []
            print('task_idx[0]==[1, 0]', test_task_idx[0][0] == 0, test_task_idx[0][1] == 1)
            for i in range(len(test_task_idx)):
                if test_task_idx[i][0] == 1:
                    hs_pred.append(test_pred_label[i])
                    hs_y.append(test_y_label[i])
            print('test set size=',len(hs_y))
            return hs_pred, hs_y

        if self.model_type in {'CNN'}:
            loss, acc = model.evaluate(self.test_chars, self.test_y, batch_size=self.batch_size, verbose=0)
            self.test_pred = model.predict(self.test_chars, batch_size=self.batch_size)
        elif self.model_type in {'HHMM_transformer'}:
            loss, acc = model.evaluate([self.test_x, self.task_idx_test, self.ruling_embedding_test], self.test_y, batch_size=self.batch_size, verbose=0)
            self.test_pred = model.predict([self.test_x, self.task_idx_test, self.ruling_embedding_test], batch_size=self.batch_size)
        else:
            loss, acc = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size, verbose=0)
            self.test_pred = model.predict(self.test_x, batch_size=self.batch_size)
        
        self.test_pred_label = np.argmax(self.test_pred, axis=1)
        self.test_y_label = np.argmax(self.test_y, axis=1)
        hs_pred, hs_y = get_hateSpeech(self.task_idx_test, self.test_y_label, self.test_pred_label)
        # hs_pred, hs_y = self.test_pred_label, self.test_y_label
        self.f1_hs_wei = f1_score(hs_pred, hs_y, average='weighted')
        self.f1_hs = f1_score(hs_pred, hs_y, average='macro')
        # self.auc = roc_auc_score(self.test_y_label, self.test_pred[:,1])
        self.acc = accuracy_score(hs_pred, hs_y)
        self.f1_all = f1_score(self.test_y_label, self.test_pred_label, average='macro')
        self.report = classification_report(hs_pred, hs_y)


        if self.f1_hs >= self.best_test_f1:
            self.best_test_f1 = self.f1_hs
            self.best_test_epoch = epoch
            self.best_report = self.report
            # model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)

        if print_info:

            logger.info("Evaluation on test data: loss = %0.6f accuracy = %0.2f%%" % (loss, acc * 100) )
            self.print_info()

    def print_info(self):
        logger.info("Evaluation on test data: acc = %0.6f " % self.acc)
        logger.info("Evaluation on test data: f1_hs = %0.6f " % self.f1_hs)
        logger.info("Evaluation on test data: f1_hs_wei = %0.6f " % self.f1_hs_wei)
        # logger.info("Evaluation on test data: auc = %0.6f " % self.auc)
        logger.info("Evaluation on test data: f1_all = %0.6f " % self.f1_all)
        logger.info('--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        logger.info('--------------------------------------------------------------------------------------------------------------------------')
        logger.info('Best @ Epoch %i:' % self.best_test_epoch)
        logger.info('BestF1 %0.6f ' % self.best_test_f1)
        logger.info('  [TEST] report %s' % self.best_report)
