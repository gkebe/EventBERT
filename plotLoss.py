# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:24:15 2020

@author: T530
"""
import re
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
    def loss_plot(filename):
        f = open(filename, "r", encoding="utf8")
        text = f.read()
        
        iterations = re.findall("Training Iteration: [\d]+",text)
        iterations = [int(re.findall("[\d]+", i)[0]) for i in iterations]
        
        epochs = re.findall("Training Epoch: [\d]+",text)
        epochs = [int(re.findall("[\d]+", i)[0]) for i in epochs]
        
        losses = re.findall("average_loss: [\d\.]+",text)+re.findall("final_loss: [\d\.]+",text)[1:]
        losses = [float(re.findall("[\d\.]+", i)[0]) for i in losses]
        for i in range(len(iterations)):
            print(str(iterations[i]*81920)+","+str(losses[i]))
        plt.plot(epochs,losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs '+os.path.split(filename)[1].replace(".log",""))
        plt.legend()
        if not os.path.exists("results/plots"):
            os.makedirs("results/plots")
        plt.savefig('results/plots/Loss vs Epochs '+os.path.split(filename)[1].replace(".log","")+".png", format='png', dpi=300)
        plt.close()
        plt.plot(iterations,losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss vs Iterations '+os.path.split(filename)[1].replace(".log",""))
        plt.legend()
        plt.savefig('results/plots/Loss vs Iterations '+os.path.split(filename)[1].replace(".log","")+".png", format='png', dpi=300)
        plt.close()
    loss_plot(args.filename)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--filename",
                        type=str,
                        required=True,
                        help="Specify a input filename!")
    args = parser.parse_args()
    main(args)