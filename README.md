# EC523_MusicGeneration
Final project for Spring 2023 EC523 involving music generation task using WaveNet model

### Team members: 

Sally Shin, Yuke Li, William Krska

### Link to Google Drive

https://drive.google.com/drive/u/1/folders/0AEwRQcK0vvTPUk9PVA

Exploring Piano Music Generations Using Various Architectures
Sally Shin, William Krska, Yuke Li 
salshin@bu.edu, wkrska@bu.edu, yukeli@bu.edu 





1. Task
For this project, we’ll be generating piano music using different models, 1) simplified WaveNet + LSTM model using chords from Bach compositions, 2) RoBERTa and 3) GPT models using tokenized MIDIs from MAESTRO dataset. We aim to generate and evaluate piano music that sounds natural on inspection and indistinguishable from human-made piano music, in 5-10 second intervals. The goal is to produce performances that are musically plausible and coherent, and that capture the style and characteristics of the training data. We will also evaluate our generated music using several metrics, including note accuracy, rhythmic accuracy, pitch coherence, and human evaluation.

2. Related Work
“Wavenet: A Generative Model For Raw Audio” by Aaron van den Oord et. al.
In this work, the authors introduce the WaveNet, a fully probabilistic and autoregressive model that produces a predictive audio distribution that takes into account all previous audio. The authors describe the general architecture of the model to be a series of convolutional layers without any pooling layers in between. The input length is the same as the output length. The key point for this model is that they utilize causal convolution to reduce the computational load and time in training. To reduce the number of layers needed, the authors also utilize dilated convolutions to increase the receptive field by implementing a coarser scale than regular convolutions.  

“Bach Genre Music Generation with WaveNet—A Steerable CNN-based Method with Different Temperature Parameters” by S Luo
In this work, the authors propose a novel method for generating Bach-style music using WaveNet, a deep neural network architecture originally developed for speech synthesis. They used 382 chorales as the dataset, and they treated chords (4 notes) of each time step as arpeggios so the model could predict all four notes in turn. For music generation, the authors use a technique called "temperature sampling" to control the degree of randomness in the generated music, and they show that different temperature settings can produce music with different characteristics. They retained 10 chorales as the seed chords for generating Bach’s style music and the generated pieces were evaluated by professional and non-professional listeners. 

“Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions” by Huang and Yang. 

In this work, authors use a customized Byte-Pair Encoding tokenizer to tokenize pop music MIDI files. Their dataset derives from popular music covers on the Internet in audio format and changed to MIDI format using “Onsets and Frames” method. After converting to MIDI representation, token IDs are generated using MIDI-specific events and changed to Byte-Pair Encoding tokenization. The tokenization of MIDI files using ‘REMI’ tokenizer is significant in that it explores a new, event-based representation than previously used ‘MIDI-like’ representation. REMI tokenizer has several novel MIDI events that allows for easier understanding of the encoded music. It notably introduces note duration events instead of note-on and note-off events and also position and bar events to make the time scale of the music more apparent. Our own dataset preparations use the authors’ tokenizer for our MIDI dataset. 
“RoBERTa: A Robustly Optimized BERT Pretraining Approach” by Yinhan Liu et al. 
In the work, the authors present a new method for pretraining large-scale language models called RoBERTa, which is an extension of the BERT (Bidirectional Encoder Representations from Transformers) model. RoBERTa made several modifications to the BERT training procedure that result in a more robust and accurate model. RoBERTa made several modifications to the BERT training procedure that result in a more robust and accurate model. They remove next sentence prediction (NLP) objective and used dynamic masking. The authors found that removing NLP objective and only training on the masked language modeling (MLM) task resulted in a better-performing model.

“Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset” by Christopher Hawthorne et al. 
In this work, the authors use the WaveNet model with MAESTRO dataset and design a state-of-the-art architecture to produce longer music sequences that sound natural. The factorized hierarchical model separates the musical structure, timing, and dynamics of piano performances into different factors, which are modeled independently at each level. This allows for greater flexibility in generating new piano performances, as different factors can be combined in different ways. The authors evaluate the music based on musicality, coherence, and expressiveness. A user survey was also conducted. 

“DeepJ: Style-Specific Music Generation” by Huanru Henry Mao et. al.
In this work, the authors look to improve on previous efforts to create a deep neural network that can compose music in a specific style. Prior architectures such as the Biaxial LSTM used a more simplistic method of representing notes, only storing their “on” or “off” values. The authors, still using a biaxial LSTM, wanted to make the music tunable, and include more aspects of music such as dynamics to better mimic human created music. The notes are stored in an NxT matrix, where N is the number of possible notes, and T is that notes value at a given time sample.

3. Approach
For our approach, we chose to implement three different architectures, one using the Bach chorales dataset and the other two using MAESTRO MIDI dataset. We initially started with the Bach chorale dataset using simplified WaveNet attached to an LSTM layer to familiarize ourselves with the task. We then moved on to using more complex dataset and Transformer architecture to produce a more complex piano music. 

3.1 Bach Chorale and WaveNet + LSTM
We start by reimplementing “Bach genre music generation with WaveNet” by S Luo. We found the same dataset used in the paper and reimplemented their pipeline based on the descriptions as best we could. 

Figure _. Architecture of model from S Luo [_]

The paper describes this architecture as a simplified WaveNet, which removes the residual and skip connections present in the original WaveNet. Since the input data used for this implementation is a much simpler representation of music, where we only feed in 4 pitch notes per chord at a set time signature across the entire dataset, this model does not need to utilize all of the parts of the original WaveNet. 

This model consists of an embedding layer, which maps each note to a continuous vector representation, followed by a simplified WaveNet with layers of 1D causal convolution layers with increasing dilation. Each CNN later is followed with a BatchNormalization layer to capture the temporal dependencies of the music. The output of the WaveNet is then passed through an LSTM layer to capture longer-term patterns in the music, and a dense layer with a SoftMax activation is used to predict the probability distribution over the notes at each time step. 

During training, the model is optimized to minimize the cross-entropy loss between the predicted probability distribution and the ground-truth distribution. During generation, the model uses temperature sampling to control the degree of randomness in the generated music, which allows the user to adjust the "creativity" of the model and produce music with different characteristics. Overall, the model architecture and training procedure are designed to generate high-quality Bach-style music that is both faithful to the original style and creative in its own right.

After we familiarized ourselves with the pipeline for generating music pieces, we moved on to work with (Musical Instrument Digital Interface) MIDI representation. MIDI data consists of a series of events, such as note on/off messages, velocity, and duration, which can be used to represent a wide range of musical information.


3.2 Tokenization of MIDI representation
To tokenize the MIDI dataset, we used a custom implementation of REMI tokenizer provided in MidiTok package. The REMI tokenizer works by parsing a MIDI file and converting it into a sequence of tokens. Each token represents a musical event, such as a note or rest, and contains information about the timing, duration, and pitch of the event. We include tokens to represent various bar, position, pitch, velocity and duration. We also include the special tokens <bos> and <eos> to indicate beginning and end of a song, <pad> to indicate padding, and <mask> to indicate masking. Since piano piece has variable length, we set the maximum length to be the length of longest song and add <pad> token to the end of the every sequence until it reaches the maximum length. We include <mask> token for masked language modeling (MLM) used in RoBERTa training. When a <mask> token is inserted in the phrase, it indicates that the model should predict the missing token in the phrase.

Figure

3.3 MAESTRO and RoBERTa





3.4 MAESTRO and GPT-mini
For our second architecture using our tokenized MAESTRO dataset, we chose a mini GPT model with 12 layers, 12 number of heads, and embedding length of 192 tokens. In total, it has 5.47M trainable parameters. GPT model is set up as a series of Transformer blocks that utilize multi headed self-attention that allows for self-attention module to run parallel for efficient computing. Layer normalization is added to prevent gradients from vanishing or exploding during training. 
 
Figure _. Layers within a Transformer Block

For our input data to the model, we use the tokenized MAESTRO dataset and give the model initial input of the embedding length and calculate loss by having the model predict the next logit in the sequence and comparing to the actual value from the training dataset. We used cross entropy to calculate the loss. We trained until the loss plateaued at around epoch 30 and saved the model with lowest loss during training. 
To generate music with the trained model, 

General summary of 3 architecture approach
Bach Chorales
Simple, chord representation of 387 Bach chorales with same time signatures
Re-implemented work from S Luo on using the same dataset and coded for a simplified WaveNet attached to LSTM
Mention architecture details
And how the music was generated after training & its length
RoBERTa
Short description of model
Pretrained from huggingface
Using tokenized MAESTRO dataset 
Mention masking percentages/ masking approach
Any other model details
How the music was generated after training & length of music produced
GPT
Used standard implementation with multihead self-attention and transformer blocks
Using tokenized MAESTRO dataset
Approach: to give embedding length of input, predict the following token, calculate loss using cross entropy
Layers, embedding length, number of parameters
Music was generated after training using an embedding length of input as seed and successively producing the next predictions after feeding back the previously predicted token - for a set amount of tokens - generally double the original embedding length input

4. Dataset(s)
For our dataset, we used MAESTRO version 3.0.0, a dataset that is composed of about 200 hours of piano performances. These audio recordings have been meticulously labeled with the appropriate MIDI note values, which are each composed of pitch, velocity, onset time, and offset time. A training, validation, and testing set are split into 5.66, 0.64, and 0.74 million notes respectively. For our RoBERTa and GPT architectures, we used the MIDI representations with 961 songs in the training, 137 in validation, and 177 in testing sets. 

For our initial implementation, we used the Bach chorales dataset, which took advantage of regularized musical aspects of 327 Bach chorales, which all have the same tempo and same time signatures. The chorales were transformed from sheet music into chord matrices that contain 4 pitches every row, where each row represents 1 chord. 
(S Luo [6])


5. Evaluation Metrics
The artistic nature of music, and the subjectivity that is inherent to evaluating musicality, present a unique challenge to objectively evaluating the quality of generated music. We have adopted a statistical approach to compare certain quantitative representations of the generated tracks against the training tracks. 
For each track, we calculated the following: 
pitch_range: The difference between the highest pitch and lowest pitch in a track, as encoded in MIDI. Can range from 0-127.
n_pitches_used: The number of unique pitches used, where the same note on different octaves are counted separately. Can range from 0-127.
n_pitch_classes_used: The number of unique pitch classes used, which are defined as the same note on different octaves (i.e. C5 and C7 share a pitch class). Can range from 0-12.
polyphony: The average number of pitches being played at any given time. Can range from 0-127.
polyphony_rate: The proportion of time where at least 2 pitches are concurrently played. Can range from 0-1.
scale_consistency: The proportion of time a single musical scale or key is used. Can range from 0-1.
pitch_entropy: Shannon’s entropy of pitches in the track.
pitch_class_entropy: Shannon’s entropy of pitch classes in the track.
These metrics classes of the training dataset are then filtered by how prominent a statistical distribution is present in them. We perform the Shapiro-Wilk’s test with a low threshold of 𝛼=1e-19 to find metrics with a roughly Gaussian distribution with which we will compare the metrics of generated music.
We perform a t-test with a 95% confidence interval on the relevant metrics for the training and generated data.

6. Results
The various permutations of the RoBertA architecture were able to generate several hundred tracks of roughly 150-900 seconds in duration. The training loss stabilizes after only a few hundred iterations, and does not significantly improve after several thousand iterations. 
The GPT model generates only a few tracks of roughly 30 seconds in duration. The training loss does not decrease significantly after numerous epochs. 
Below is a summary of the performance metrics for each of the models. For each of the RoBertA models, a large portion of the generated tracks passed at least four of the five valid evaluation metrics, those being pitch entropy, pitch class entropy, polyphony, and scale_consistency. However, the average polyphony rate fell well below that of the training data, at around 0.17 versus 0.85.

Model
Average Metrics Passed
Prop. ≥4 metrics passed
RoBertA 50
3.43 of 5
65.69% (90/137)
RoBertA 30
3.84 of 5
78.83% (108/137)
RoBertA 1024 50
3.24 of 5
56.2% (77/137)
RoBertA 1024 30
3.69 of 5
73% (100/137)
GPT
2.58 of 5
33.33% (13/39)

Table 1: Performance of models on Maestro Dataset

Model
Average Metrics Passed
Prop. ≥5 metrics passed
Wavenet_top1
5.05 of 6
95.16% (59/62)
Wavenet _top3
5.05 of 6
95.16% (59/62)
Wavenet _top5
5.05 of 6
95.16% (59/62)
Wavenet _top10
3.94 of 6
38.7% (24/62)
Wavenet _top30
2.73 of 6
8.1% (5/62)

Table 2: Performance of models on Bach Chorales Dataset


7. Conclusion
RoBERTa performed well in most metrics when trained on the Maestro dataset, and its performance improved with less masking as was expected. The only obvious shortcoming was the significantly lower polyphonic rate, yielding simpler sounding music than the source data. However, subjectively the RoBERTa results had a distinctively musical quality to them. 
WaveNet similarly performed very well on the Bach Chorales dataset when a reasonable temperature was used. All top 1, top 3, and top 5 selection configurations performed nearly identically. It was only after nearing the practical limit and extending to top 10 and top 40 that the results began to significantly break down. 

RoBERTa
Overall good, better with less masking as expected
WaveNet
WaveNet performed very well in our evaluation pipeline
Higher number of top choices to select from, less it passed the evaluation
Which is great, shows that the evaluation metrics show musical aspects from the original dataset
GPT
dunno

Appendix A: Team Contributions
(table 1)

Appendix B: Code Repository
GITHUB: https://github.com/sassmander/EC523_MusicGeneration

References
“Enabling factorized piano music modeling and generation with the Maestro dataset” by Hawthorne et al. 2019. https://arxiv.org/pdf/1810.12247.pdf
“WaveNet: A Generative Model for Raw Audio” by Oord et al. 16 Sep 2016. https://arxiv.org/pdf/1609.03499.pdf
“On the evaluation of generative models in music” by Yang and Lerch. 2018. https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2018/11/postprint.pdf
“DeepJ: Style-Specific Music Generation” by Mao et al. 2018. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8334500 
“Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions” by Y Huang and Y Yang. 10 Aug 2020. https://arxiv.org/pdf/2002.00212.pdf
“Bach Genre Music Generation with WaveNet—A Steerable CNN-based Method with Different Temperature Parameters” by S Luo. 10 Aug 2022. https://doi.org/10.1145/3568923.3568930
Evaluation Metrics: https://github.com/mkornyev/PyMusic
Tokenizer: https://miditok.readthedocs.io/en/v2.0.3/index.html
GPT model: https://github.com/pier-maker92/bachsformer
RoBERTa: https://huggingface.co/docs/transformers/model_doc/roberta






Appendix A: Team Contributions

Name
Task
File names
No. lines of code
Will Krska






Yuke Li






Sally Shin








