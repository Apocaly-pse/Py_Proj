from turtle import *
# 设置线条的颜色以及填充颜色
color('red', 'yellow')
# 设置绘画的速度，0为最快
speed(0)
# 开始绘制
begin_fill()
while True:
    # home()
    # print(pos())
    fd(200)
    circle(20)
    lt(170)
    # setpos(200, 200)
    fd(180)
    circle(30)
    lt(150)
    # print(pos())
    # print(abs(pos()))
    if abs(pos()) < 1:
        break
# 结束绘制
end_fill()
# 停止绘制，不退出窗口
done()
