# Progetto di introdizzione all'Apprendimento Automatico
 _a.a. 2023/2024_ 
## Deblurring di immagini mosse
Il progetto riguarda il deblurirng di immagini. Il tipo di "blur" in questione è ottenuto mediante sovrapposizione di immagini leggermente traslate a partire dall'originale, lungo un tragitto randomico. Il blur intende simulare il tremolio su immagini con tempi di esposizione prolungati.\
usimao mnist come dataset.\
**generatore:**
```python
def generator(dataset,batchsize,moves=10):
  while True:
    rand = np.random.randint(dataset.shape[0],size=batchsize)
    y_truth = dataset[rand]
    blurred = np.copy(y_truth)
    moving = tf.expand_dims(y_truth,axis=-1)

    for i in range(moves):
      #RandomTranslation requires a channel axis
      moving = layers.RandomTranslation(0.07,0.07,fill_mode='constant',interpolation='bilinear')(moving)
      blurred = blurred + tf.squeeze(moving)
    blurred = blurred/(moves+1)
    yield(blurred,y_truth)

```
## valutazione 
per valutare il modello usiamo il Mean Square Error (mse) medio 