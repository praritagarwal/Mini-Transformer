Project to implement a mini version of the Transformer as described in [this paper](https://arxiv.org/abs/1706.03762). I also found Jay Alammar's [blog](http://jalammar.github.io/illustrated-transformer/) quite helpful in understanding the various concepts involved. 

I tried to implement this version of transformer without looking at anyone's else's implementation and finally compared its performance to that of fastai's implementation on a French-to-English translation task. 

The repository contains 2 jupyter notebooks and a python module which are as follows:
  
  1)[Transformer.py](Transformer.py): This contains my implementation of the Transformer class.
  
  2)[MiniTransformer.ipynb](MiniTransformer.ipynb): This contains the various test and experiments I ran on different pieces of code that I wrote as part of my implementation of the Transformer class. As such this can be considered the notebook equivalent of my scratch pad.

  3)[French-to-English-Transformer.ipynb](French-to-English-Transformer.ipynb): In this notebook we import the Transformer class and train it to translate sentences from French to English. 

The dataset we used is a subset of the dataset created by Chris Callison-Burch and provided [here](http://www.statmt.org/wmt15/translation-task.html). More specifically, we extracted samples which appeared in the form of questions. 

Following are a couple of translations obtained from our model:

1) 

     French Sentence: 'xxbos quelles méthodes a - t - on trouvées particulièrement efficaces pour consulter le public et les parties intéressées sur la protection des renseignements personnels reliés à la santé ?'

     Target Sentence: 'xxbos what approaches have been found particularly effective in consulting with the public and stakeholders on the protection of personal health information ?'

     Transformer Output: 'xxbos what methods have been found particularly effective in addressing the the public and stakeholders on the protection of personal health information ?'


2) 

     French Sentence: "xxbos qui a le pouvoir de modifier le règlement sur les poids et mesures et le règlement sur l'inspection de l'électricité et du gaz ?"

 
     Target Sentence: 'xxbos who has the authority to change the electricity and gas inspection regulations and the weights and measures regulations ?'

     Transformer Output: 'xxbos who has the authority to change the regulations and control measures measures ? to regulations and regulations of ?'

3) 

     French Sentence: 'xxbos ´ 'ou sont xxunk leurs grandes convictions en ce qui a trait a la '' ´ transparence et a la responsabilite ?'
 
     Target Sentence: 'xxbos what happened to their great xxunk about transparency and accountability ?'
 
     Transformer Ouput: 'xxbos what are to them views beliefs beliefs transparency and accountability ?'


 4) 

      French Sentence: 'xxbos de quoi l’afrique a - t - elle vraiment besoin pour se sortir de la pauvreté ?',
 
      Target Sentence: 'xxbos what does africa really need to pull itself out of poverty ?',
 
      Transformer Output: 'xxbos what does africa really need to pull out out of poverty ?'

 5)

      French Sentence: 'xxbos quelles ressources votre communauté possède - t - elle qui favoriseraient la guérison ?',
 
      Target Sentence: 'xxbos what resources exist in your community that would promote recovery ?',

      Transformer Ouput: 'xxbos what resources does in your community that would help healing ?'




