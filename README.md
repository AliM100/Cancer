# Cancer
First install packages required in requirements.txt file
## For training :
 -you need to split data (original data file: https://web.inf.ufpr.br/vri/breast-cancer-database/) using command:
 ```
 python main.py preprocess()
 ```
 -after preprocessing start training model : 
  ```
  python main.py train()
  ```
## For testing:
 -Download weight files if you want to test the model. Weight files: https://drive.google.com/file/d/1kIScu4xqv5dmuI5SnW3xNZ0Ou0CcVITP/view?usp=share_link
 -Run testing command:
 ```
 python main.py test()
 ```
