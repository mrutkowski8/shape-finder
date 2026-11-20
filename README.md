# **PURPOSE**

#### this project is made to recognize whether you draw:

* a circle
* a square
* a triangle


# **REQUIREMENTS**

#### this project will not run without the following libraries:

* PyTorch ( torch )
* Pillow ( PIL )
* tkinter ( tkinter )



# **TRAINING**

### put labeled PNGs in samples with names such as circle1.png, square1.png, triangle1.png; then run:

```python
python train.py
```

training the model runs on the CPU and saves the learned weights to shape_model.pt
app then loads those weights when it starts and shows the model's scores for the three shapes live each 50ms



# **RUN**
### to run the program you have to run:

```python
python main.py
```


*this is my first achievable project ever so spread no hate and i hope you have fun testing it out! :)*

