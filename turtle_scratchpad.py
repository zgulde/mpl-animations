import itertools as it
from zgulde import take
from random import choice, randint
import turtle


def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


take(fib(), 10)


turtle.color("blue", "green")

turtle.begin_fill()

while True:
    turtle.forward(200)
    turtle.left(170)
    if abs(turtle.pos()) < 1:
        break

turtle.end_fill()

turtle.clearscreen()
turtle.begin_fill()
turtle.speed(10)
for n in it.islice(fib(), 100):
    turtle.forward(n)
    turtle.left(5)
    if abs(turtle.pos()) > 400:
        turtle.up()
        turtle.setx(0)
        turtle.sety(0)
        turtle.down()

take(fib(), 100)
