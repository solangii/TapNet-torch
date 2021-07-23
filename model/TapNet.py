from model.model_ResNet12 import EmbeddingNet

class TapNet:
    def __init__(self, config, dataloader, exp_name):
        self.config = config
        self.dataloader = dataloader

        self.device = config.device
        self.n_class_train = config.n_class_train
        self.n_class_test = config.n_class_test
        self.input_size = config.input_size
        self.dim = config.dim

        self.EmbeddingNet = EmbeddingNet(self.dim, self.n_class_train).to(self.device) #파라미터 왜필?

    def Projection_Space(self):
        return None

    def compute_power(self):
        return None

    def compute_power_avg_phi(self):
        return None

    def compute_loss(self):
        return None

    def compute_accuracy(self):
        return None

    def select_phi(self):
        return None

    def train(self):
        

        return None

    def evaluate(self):
        return None

    def decay_learning_rate(self, decaying_parameter=0.5):
        return None


def train(model, config, dataloader, exp_name):
    loss_h = []
    accuracy_h_val = []
    accuracy_h_test = []

    acc_best = 0
    epoch_best = 0

    for idx, episode in enumerate(dataloader['meta_train']):
        support_data, support_label = episode['train']
        query_data, query_label = episode['test']
        support_data, support_label = support_data.to(config.device), support_label.to(config.device)
        query_data, query_label = query_data.to(config.device), query_label.to(config.device)

        loss = model.train(support_data, support_label) # 맞나

        # logging
        # --------------------------------
        loss_h.extend([loss.tolist()])
        if idx % 50 ==0:
            print("Episode: %d, Train Loss: %f "%(idx, loss))

        if idx!=0 and idx%500 ==0:
            print("Evaluation in Validation data")
            scores = []

            for idx, episode in enumerate(dataloader['meta_val']): #몇개 돌지 정해두기
                accs = model.evaludate(support_data, support_label)
                accs_ = [cuda.to_cpu(acc) for acc in accs] # 이거머지
                score = np.asarray(accs_, dtype=int) # 이거머지
                scores.append(score)

            print(('Accuracy 5 shot ={:.2f}%').format(100*np.mean(np.array(scores))))
            accuracy_t = 100*np.mean(np.array(scores))

            if acc_best < accuracy_t:
                acc_best = accuracy_t
                epoch_best = idx
                # save model Todo
                # 뭐시l save npz

            accuracy_h_val.extend([accuracy_t.tolist()])
            del(accs) # 이거머지
            del(accs_)
            del(accuracy_t)

        if idx!=0 and idx%config.lrstep==0 and config.lrdecay:
            model.decay_learning_rate(0.1)

def eval(model, config, dataloader):
    accuracy_h5 = []

    #load model

    print("Evaluating the best 5shot model...")
    for i in range(50):
        scores =[]
        for idx, episode in enumerate(dataloader['meta_test']):
            support_data, support_label = episode['train']
            query_data, query_label = episode['test']
            support_data, support_label = support_data.to(config.device), support_label.to(config.device)
            query_data, query_label = query_data.to(config.device), query_label.to(config.device)

            accs = model.evaluate(support_data, support_label)
            accs_ = [cuda.to_cpu(acc) for acc in accs] #이거머지
            score = np.asarray(accs_, dtype=int)
            scores.append(score)
        accuracy_t = 100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
        print(('600 episodes with 15-query accuracy: 5-shot = {:.2f}%').format(accuracy_t))

        del(accs)
        del(accs_)
        del(accuracy_t)

        #sio.savemat(savefile_name, {'accuracy_h_val':accuracy_h_val, 'accuracy_h_test':accuracy_h_test, 'epoch_best':epoch_best,'acc_best':acc_best, 'accuracy_h5':accuracy_h5})

    print(('Accuracy_test 5 shot ={:.2f}%').format(np.mean(accuracy_h5)))




