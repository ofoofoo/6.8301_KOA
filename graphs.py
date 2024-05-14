import numpy as np
import matplotlib.pyplot as plt



def draw_graph(arr):
# Load the data from the .npy file
    for str in arr: 
        data = np.load("./results/cifar10/"+str[0]+str[1]+"_test_loss.npy")
        
    # Create x values from 1 to 20 (as there are 20 numbers)
        x_values = np.arange(1, 21)

    # Plot the data as a line graph
        plt.plot(x_values, data[:20], color=str[3], label=str[2])

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss with Epoch on CIFAR-10 or various models')
    plt.xticks(np.arange(1, 21))
    plt.legend()
    # Show the plot
    plt.show()

#file name, legend name, color

arr=[["depth_scaling/", "cifar10_6_KAN", "2^6 times Depth Scaled KAN", "blue"], 
     ["width_scaling/", "cifar10_1024_KAN", "2^6 times Width Scaled KAN", "red"],
     ["experts_scan/", "cifar10_4_MOEKAN", "MoEKAN with 4 experts", "green"],
     ["experts_scan/", "cifar10_9_MOEMLP", "MoEMLP with 9 experts", "orange"],
    #  ["width_scaling/", "", "MLP", "teal"],
     ["mixers/","S4KANMixer", "KANMixer","magenta"],
     ["mixers/","L4MLPMixer", "MLPMixer","pink"]
        ]
draw_graph(arr)