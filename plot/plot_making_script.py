import numpy as np
import matplotlib.pyplot as plt

vanilla_model = "200d_1x128LSTM_2x128out_50drop.txt"
pos_model = "200d_50pos_1x128LSTM_2x128out_50drop_2.txt"

def main():
    data_vanilla = load_data(vanilla_model)
    data_pos = load_data(pos_model)
    
    print(data_vanilla.shape)
    print(data_pos.shape)
    show_plot(data_vanilla, data_pos)
    
def load_data(path):
    # Return [None, 4]
    # 0:Train loss
    # 1:Train acc
    # 2:Valid loss
    # 3:Valid acc
    file = open(path)
    
    data = []
    for line in file:
        data.append(str_to_num(line))
        
    return np.array(data)
        
def str_to_num(str):
    # Gets a string containing numbers separated by space and returns a 
    # list of floats.
    return [float(num_str) for num_str in str.split()]
    
def show_plot(data_v, data_p):
    # Accuracy plot
    plt.subplot(2,1,1)
    plt.plot(data_v[:,-1], label="Vanilla")
    plt.plot(data_p[:,-1], label="POS model")
    plt.ylabel("Validation accuracy")
    plt.legend(loc="lower right")
    
    plt.subplot(2,1,2)
    plt.plot(data_v[:,2], label="Vanilla")
    plt.plot(data_p[:,2], label="POS model")
    plt.ylabel("Average validation cross entropy")
    plt.xlabel("Iterations")    
    plt.legend(loc="upper right")
    plt.show()
    return None
        
main()