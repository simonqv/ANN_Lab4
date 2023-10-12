from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    '''
    print ("\nStarting a Restricted Boltzmann Machine..")

    # Task 1: plot convergence for different values ndim_hidden 200-500
    # OBS! if you just set  hidden_node_list = [500], you will get the weight visualisation plot
    hidden_node_list = [500, 400, 300, 200] #[200, 500]
    y = []
    plt.figure()
    for number_of_hidden_nodes in hidden_node_list:
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                        ndim_hidden=number_of_hidden_nodes,
                                        is_bottom=True,
                                        image_size=image_size,
                                        is_top=False,
                                        n_labels=10,
                                        batch_size=20
        )
        rbm.cd1(visible_trainset=train_imgs, n_iterations=10000)#30000)
        y.append(rbm.reconstruction_loss)
        print("RBM WEIGHTS TRUE SHAPE", rbm.weight_vh.shape)
        print("RBM WEIGHTS", " -5: ", len(rbm.weight_vh[rbm.weight_vh<-5])," 5: ",len(rbm.weight_vh[rbm.weight_vh>5]))
        print(rbm.weight_vh[rbm.weight_vh<-5][0:100])

        print("AVG RECON LOSS", rbm.iterations)
    
    for i, nhidden_dim in enumerate(hidden_node_list):
        plt.plot(rbm.iterations, y[i], label=f"{nhidden_dim} hidden nodes")
        plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("Average Reconstruction Error")
    plt.title("Convergence")

    plt.show()
    '''
    ''' deep- belief net '''

    print("train lbls", train_lbls)
    print(train_lbls.shape)
    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)
    
    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
    
    ''' fine-tune wake-sleep training '''
    '''
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
    '''