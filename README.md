# Subband Music Separation

This is a system for voice and accompaniment separation model training. All you need to do is configure environment, download training data and enjoy your training! If you need personalize your training, you just need to modify the configuration file (json), we also provide many examples. This repo also integrate the subband decomposation and synthesis tools mentioned in our paper: [link](https://haoheliu.github.io/Channel-wise-Subband-Input/resources/paper/Paper-Channel-wise%20Subband%20Input%20for%20Better%20Voice%20and%20Accompaniment%20Separation%20on%20High%20Resolution%20Music.pdf) 

- You can use MUSDB18 **[1]** as well as the data you have

- You can easily try  Channel-wise subband (CWS) input **[2]** by modify the configuration file

- You can use model we pre-defined. You can also add model if you like!

  ...

![subband](./pics/subband.png)

## Quick start

- First we configure the code running environments

  ```shell
  git clone https://github.com/haoheliu/Subband-Music-Separation.git
  pip install -r requirements.txt
  ```

- Then we download musdb18-hq data from the following website, this process might be a little bit slow.

  >  https://zenodo.org/record/3338373#.Xu8CqS2caCM

  Note that you can also use your own data.

- Next we configure the path to MUSDB18 dataset

  - We open the config.json (or other config file you like)
  - Modify the "MUSDB18_PATH" variable
    - e.g.: "/home/work_nfs/hhliu/workspace/datasets/musdb18hq_splited"
  - Save file

- Finanlly let training! 

  ```shell
  python main_separation.py config.json
  ```

## Demos

- **Load pre-trained model and start training(MMDenseNet)**

  - Configure model structure (Already done in config_demo_pretrained.json)

  - Configure pretrained model path (Already done in config_demo_pretrained.json)

    - ```
      "PRE-TRAINED": {
        "start_point": 155700,
        "load_model_path": "./checkpoints/1_2020_5_8_MDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs16-1_fl1.5_ss4500.0_87lnu4fshift8flength32drop0.1split_bandTrue_8"
      },
      ```

  - run and enjoy! 

    - ```shell
      python main_separation.py config_demo_pretrained.json
      ```

- **Separate a song (Using MMDenseNet)**

  - ```shell
    python demo_separation.py
    ```

  - You can also put the song you'd like to split in: "./evaluate/listener_todo"

- **Use additional data**

  - Do configuration like this:

    ```
        "additional_data": {
          "additional_vocal_path": ["addtional_data/accompaniment_list1.txt",
                                    "addtional_data/accompaniment_list2.txt"], // or more
          "additional_accompaniments_path": ["addtional_data/vocal_list1.txt",
                                             "addtional_data/vocal_list2.txt"] // or more
        }
    ```

  - List all the path to each of your .wav file in those txt files.

- **Try different kind of Channel-wise subband inputs**

  - ```
    "SUBBAND": {
      "number": 8 // you can also choose 1(no cws),2,4
    },
    ```

## About the config file

You just need to mofidy the configuration file to personalize your experiment. This section I will lead you around the variable inside the configuration file.

```
{
  "LOG": {
    "show_model_structure": 1, // 1: print model structure; 0: not print
    "every_n": 10 // print average loss every "every_n" batches
  },
  "SUBBAND": {
    "number": 8 // how many subband you want in CWS
  },
  "MODEL": {
    "PRE-TRAINED": { // optional, pre-trained model path, see my example below
      "start_point": 0, 
      "load_model_path": ""
    },
    "sources": 2, // voice and accompaniment
    "model_name": "Unet-6" // choice: ["Unet-5","Unet-6","MDenseNet","MMDenseNet"]
  },
  
  "PATH": {
    "MUSDB18_PATH": "/home/work_nfs/hhliu/workspace/datasets/musdb18hq_splited", // path to musdb18hq dataset
    "additional_data": { // optional, additional data infos
      "additional_vocal_path": [],
      "additional_accompaniments_path": []
    }
  },
  "TRAIN": {
    "device_str": "cuda:2", // device, ['cpu','cuda','cuda:1',...]
    "dropout": 0.2, // dropout_rate
    "epoches": 3000, // one epoches means 10 hours of training data
    "accumulation_step": 10, // gradient accumilation, every "accumulation_step" we update the parameters of the model. Larger "accumulation_step" equal to bigger batchsize to some sense.
    "frame_length": 3, // input frame length
    "batchsize": 4, 
    "learning_rate": { // exponential decay
      "gamma_decrease": 0.97, // decay rate
      "initial": 0.001 // initial learning rate
    },
    "loss": [ // loss function, choice: ['l1','l2','l3'] or ['l2','l3']
      "l1",   // energy conservation loss
      "l2",   // l1-norm on accompaniment
      "l3"    // l1-norm on vocal
    ]
  }
  "VALIDATION": {
    "decrease_ratio": 0.98 // If validation loss drop greater than "decrease_ratio", we save model and start evaluation on musdb18 dataset
  },
}
```



## About the training 

- We will save model in "saved_modes/"
- We will save separation results in output 
- The wav files inside "evaluate/listener_todo" are considered songs to be splitted. (separation_util.py: SeparationUtil.split_listener())

## About Channel-wise Subband (CWS) input

![tab2](./pics/tab2.png)

![tab3-sota](./pics/tab3-sota.png)

## Citation

>  [1] Z. Rafii, A. Liutkus, F.-R. St¨oter, S. I. Mimilakis, and R. Bittner, “Musdb18-hq - an uncompressed version of musdb18,” Aug. 2019.

> [2] https://haoheliu.github.io/Channel-wise-Subband-Input/