from collections import namedtuple

import pygame
from QLearning import Agent
from QLearning import load_model, save_model

from matplotlib import pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

plt.style.use('fivethirtyeight')
pygame.font.init()
pygame.init()

# Game Constants
WIN_WIDTH = 1270
WIN_HEIGHT = 700
SIDEWALK_COLOR = (112, 92, 74)
ROAD_COLOR = (122, 120, 118)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (207, 207, 207)
RED = (255, 34, 5)
ROAD_LINE_W = 80  # 105
ROAD_BOARD_W = 51
GAME_WIDTH = 375
params = False

# Agent variables
Max_size = 300000
Lr = 0.001
Gamma = 0.99
Eps_dec = 15e-4
Eps_end = 0.0
Epsilon = 1.0
INPUT_DIMS = 194

# Style
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Traffic")
icon = pygame.image.load('src/icon.png')
pygame.display.set_icon(icon)
FONT = pygame.font.Font('freesansbold.ttf', 20)

THINKING = pygame.USEREVENT + 1
CARS_SPAWN = pygame.USEREVENT + 2
DRAIWING = pygame.USEREVENT + 3
RESET_SET_PARAMS = pygame.USEREVENT + 4

pygame.time.set_timer(THINKING, 100)
pygame.time.set_timer(CARS_SPAWN, 8500)
pygame.time.set_timer(DRAIWING, 1)
pygame.time.set_timer(RESET_SET_PARAMS, 16000)

# Cars images
main_car = pygame.image.load('src/car__.png')  # width 80px
road = pygame.image.load('src/road_extended3.png')  # 630x700 pixels
car1 = pygame.image.load('src/car2__.png')  # width 80px
car2 = pygame.image.load('src/car3__.png')  # width 80px
car3 = pygame.image.load('src/car4__.png')  # width 80px
car4 = pygame.image.load('src/car5__.png')  # width 80px
car5 = pygame.image.load('src/car6__.png')  # width 80px
car6 = pygame.image.load('src/car7__.png')  # width 80px
bus = pygame.image.load('src/bus__.png')  # width 100px
bus1 = pygame.image.load('src/bus1__.png')  # width 10px


class Button:

    def __init__(self, rect=(1000, 10, 130, 30), text='', color=BLACK, thin=2, clicked_color=LIGHT_GRAY, clicked=False):
        self.size = (rect[2], rect[3])
        self.text = text
        self.x = rect[0]
        self.y = rect[1]
        self.clicked = clicked
        self.color = color
        self.border_thin = thin
        self.clicked_color = clicked_color
        self.rect = pygame.Rect(rect)
        self.text_rect = rect

        def create():
            """Create button on the screen"""
            self.text_font = FONT.render(self.text, True, self.color)
            self.text_rect = self.text_font.get_rect()
            self.text_rect.center = (int(self.x + self.size[0] / 2), int(self.y + self.size[1] / 2))

        create()

    def draw(self):
        if self.clicked == True:
            self.text_font = FONT.render(self.text, True, self.clicked_color)
        else:
            self.text_font = FONT.render(self.text, True, self.color)
        win.blit(self.text_font, self.text_rect)
        pygame.draw.rect(win, BLACK, self.rect, self.border_thin)

    def click(self):
        pass


class Cars:

    def __init__(self, num):
        self.y = 0
        self.img = num
        self.bus = False

        def random_image(number):
            if number == 1:
                self.img = car1
            elif number == 2:
                self.img = car2
            elif number == 3:
                self.img = car3
            elif number == 4:
                self.img = car4
            elif number == 5:
                self.img = car5
            elif number == 6:
                self.img = car6
            elif number == 7:
                self.img = bus
                self.bus = True
            elif number == 8:
                self.img = bus1
                self.bus = True

        random_image(self.img)

        self.height = self.img.get_height()
        self.speed = random.randint(5, 7)
        self.direction = random.randint(0, 1)

        def appear():

            if self.direction == 0:  # From up to down
                self.y = - 150
                self.flipped_img = pygame.transform.flip(self.img, False, True)
                if self.bus:
                    self.x = ROAD_BOARD_W + ROAD_LINE_W * 2 + random.randint(0, 1) * ROAD_LINE_W - 3
                else:
                    self.x = ROAD_BOARD_W + ROAD_LINE_W * 2 + random.randint(0, 1) * ROAD_LINE_W
                self.img = self.flipped_img
            else:
                self.y = 750
                if self.bus:
                    self.x = ROAD_BOARD_W + random.randint(0, 1) * ROAD_LINE_W - 3
                else:
                    self.x = ROAD_BOARD_W + random.randint(0, 1) * ROAD_LINE_W

        appear()
        self.rect = self.img.get_rect(topleft=(self.x, self.y))

    def move(self):
        if self.direction == 0:  # From up to down
            self.y += self.speed
            self.rect.y += self.speed
        else:
            self.y -= self.speed
            self.rect.y -= self.speed

    def draw(self):
        # main_car = pygame.image.load('car_.png')  # width 80px
        win.blit(self.img, (self.x, self.y))


class Car:

    def __init__(self, brain=None):  # brain=None
        self.x = ROAD_BOARD_W + ROAD_LINE_W
        self.y = 350
        self.angle = 0  # degrees to tilt
        self.img = main_car
        self.height = self.img.get_height()
        self.width = self.img.get_width()
        self.rect = self.img.get_rect(topleft=(self.x, self.y))
        self.count_anim = 0
        self.shift = (0,)
        self.hide_sensors = True


        if brain == None:
            self.brain = Agent(gamma=Gamma, epsilon=Epsilon, batch_size=10, n_actions=5, input_dims=[INPUT_DIMS],
                               lr=Lr, max_memory=Max_size)
        else:
            self.brain = brain
            global params
            if params == True:
                self.brain.epsilon = Epsilon
                self.brain.gamma = Gamma
                self.brain.mem_size = Max_size
                self.brain.eps_min = Eps_end
                self.brain.eps_dec = Eps_dec
                self.brain.lr = Lr
                params = False


        self.inter_points = []
        self.key = ''
        self.sensors = self.init_sensors()
        self.score = 0
        self.fitness = 0
        self.state = [0] * INPUT_DIMS
        self.agent_data = 0
        self.action = 4
        self.action_type = 'R'
        self.epsilon = 0.1
        self.actions = [0, 0, 0, 0, 0]
        self.loss = 0
        self.next_state = [0] * INPUT_DIMS
        self.reward = 0

    def init_sensors(self):
        car_centre = (self.x + self.width / 2, self.y + self.height / 2)
        # Sensors
        # Forehead
        fr_1 = (car_centre, (self.x + self.width / 2, self.y - 300))
        fr_4 = (car_centre, (self.x - self.width, self.y - 300))
        fr_2 = (car_centre, (self.x - self.width * 2, self.y - 300))
        fr_5 = (car_centre, (self.x + self.width * 2, self.y - 300))
        fr_3 = (car_centre, (self.x + self.width * 3, self.y - 300))
        fr_6 = (car_centre, (self.x - self.width * 3.5, self.y - 300))
        fr_7 = (car_centre, (self.x + self.width * 4.5, self.y - 300))
        # Left
        lf_1 = (car_centre, (self.x - self.width * 5, self.y - 300))
        lf_2 = (car_centre, (self.x - self.width * 5, car_centre[1]))
        lf_3 = (car_centre, (self.x - self.width * 5, self.y + self.height + 300))
        lf_4 = (car_centre, (self.x - self.width * 5, self.y - 100))
        lf_5 = (car_centre, (self.x - self.width * 5, self.y + self.height + 100))
        # Right
        rt_1 = (car_centre, (self.x + self.width * 6, self.y - 300))
        rt_2 = (car_centre, (self.x + self.width * 6, car_centre[1]))
        rt_3 = (car_centre, (self.x + self.width * 6, self.y + self.height + 300))
        rt_4 = (car_centre, (self.x + self.width * 6, self.y - 100))
        rt_5 = (car_centre, (self.x + self.width * 6, self.y + self.height + 100))
        # Back
        bk_1 = (car_centre, (self.x + self.width / 2, self.y + self.height + 300))
        bk_4 = (car_centre, (self.x - self.width, self.y + self.height + 300))
        bk_2 = (car_centre, (self.x - self.width * 2, self.y + self.height + 300))
        bk_3 = (car_centre, (self.x + self.width * 3, self.y + self.height + 300))
        bk_5 = (car_centre, (self.x + self.width * 2, self.y + self.height + 300))
        bk_6 = (car_centre, (self.x - self.width * 3.5, self.y + self.height + 300))
        bk_7 = (car_centre, (self.x + self.width * 4.5, self.y + self.height + 300))
        return [fr_1, fr_2, fr_3, fr_4, fr_5, fr_6, fr_7, bk_1, bk_2, bk_3, bk_4, bk_5, bk_6, bk_7, lf_1, lf_2, lf_3,
                lf_4, lf_5, rt_1, rt_2, rt_3, rt_4, rt_5]

    def move_up_down(self):
        # Up & Down work well
        if self.y > 0 and self.key == "up":
            self.y -= 5
            self.rect.y -= 5

        elif self.y + self.height < WIN_HEIGHT and self.key == 'down':
            self.y += 5
            self.rect.y += 5

    @staticmethod
    def move_right_left(car):
        if (car.x - ROAD_BOARD_W) % ROAD_LINE_W == 0:
            if car.key == "left" and car.x > ROAD_BOARD_W:
                if car.angle != 30:
                    car.angle = 30
                    car.img = car.rotate(car.angle)
                car.count_anim -= 1
                car.x -= ROAD_LINE_W / 10
                car.rect.x -= ROAD_LINE_W / 10

            elif car.key == "right" and car.x < ROAD_BOARD_W + ROAD_LINE_W * 3:
                if car.angle != -30:
                    car.angle = -30
                    car.img = car.rotate(car.angle)
                car.count_anim += 1
                car.x += ROAD_LINE_W / 10
                car.rect.x += ROAD_LINE_W / 10
        elif car.count_anim > 0:
            if car.angle != -30:
                car.angle = -30
                car.img = car.rotate(car.angle)
            car.count_anim += 1
            car.x += ROAD_LINE_W / 10
            car.rect.x += ROAD_LINE_W / 10
        elif car.count_anim < 0:
            if car.angle != 30:
                car.angle = 30
                car.img = car.rotate(car.angle)
            car.count_anim -= 1
            car.x -= ROAD_LINE_W / 10
            car.rect.x -= ROAD_LINE_W / 10
        if (car.count_anim == 10) or (car.count_anim == -10):
            car.img = main_car
            car.count_anim = 0
            car.angle = 0

    def draw_sensors(self, point_thickness):
        self.sensors = self.init_sensors()
        # Forehead
        if self.hide_sensors == True:
            return
        for i in range(24):
            pygame.draw.line(win, WHITE, self.sensors[i][0], self.sensors[i][1], 2)

        def draw_inter_points(point_thickness):
            if self.hide_sensors == False:
                for n in range(0, len(self.inter_points), 2):
                    if self.inter_points[n] != 0:
                        # Drawing on the screen
                        pygame.draw.circle(win, RED, (
                            int(self.inter_points[n] + point_thickness),
                            int(self.inter_points[n + 1] + point_thickness)),
                                           point_thickness)

        draw_inter_points(point_thickness)

    def detecte_intersection_points(self, cars):
        int_p = []
        data = []
        ctr = 1
        for sensor in self.sensors:
            for vehicle in cars:
                collision = line_rect_intersection(sensor,
                                                   vehicle.rect)# Calculate points of intersection for each car and sensor.
                if collision != [None, None, None, None]:
                    int_p += collision
                if len(int_p) >= 8 * ctr:
                    break
            # Every sensor must have max and min 8 intersection points with 2 cars, if it hasn't have it. It's equal to None.
            while len(int_p) < 8 * ctr:
                int_p.append(None)
            ctr += 1
        for point in int_p: # Forms suitable data for neural network.
            if point != None:
                data.append(point[0])
                data.append(point[1])
                if point[0] < 0 or point[1] < 0:
                    pass
            else:
                # if not intersect point is (0,0)
                data.append(0)
                data.append(0)
        while len(data) > (INPUT_DIMS - 2) * 2:
            data.pop(len(data) - 1)
        return data

    def collide(self, cars):
        for vehicle in cars:
            if self.rect.colliderect(vehicle.rect):
                return 1

    def rotate(self, angle):
        rotated_img = pygame.transform.rotozoom(main_car, angle, 1)
        new_topleft = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center).topleft
        self.shift = (abs(new_topleft[0] - self.x), abs(new_topleft[1] - self.y))

        return rotated_img

    def think(self, cars, flag=1):
        self.key = ''
        int_p = []
        dangerous = False
        death = False

        # SELECT ACTION
        if self.count_anim == 0:
            self.agent_data = self.brain.choose_action(self.state)
            self.action = self.agent_data['action']
            self.action_type = self.agent_data['action_type']
            self.epsilon = self.agent_data['epsilon']
            self.actions = self.agent_data['actions'].tolist()
            self.loss = self.agent_data['loss'].tolist()

        # TAKE ACTION
        self.take_action(self.action)
        # NEXT_STATE
        self.inter_points = self.detecte_intersection_points(cars)
        self.next_state = calc_distance(self, int_p=self.inter_points)
        dist = min(self.next_state)
        self.next_state.append(self.x)
        self.next_state.append(self.y)
        # REPLAY MEMORY
        if flag == 0:
            self.brain.replay_memory(state=self.state, action=self.action, reward=-10,
                                     next_state=self.next_state,
                                     flag=flag)
        else:
            self.brain.replay_memory(state=self.state, action=self.action, reward=dist / 13,
                                     next_state=self.next_state,
                                     flag=flag)
        self.state = self.next_state

    def take_action(self, action):
        if action == 0:  # Up
            self.key = 'up'
        elif action == 1:  # Down
            self.key = 'down'
        elif action == 2:  # Left
            self.key = 'left'
        elif action == 3:  # Right
            self.key = 'right'
        elif action == 4:  # None
            self.key = ''
        self.move_up_down()
        self.move_right_left(self)


from math import sqrt


def calc_distance(main_car, int_p):
    distance = []
    c = 500
    if int_p == []:
        return
    for i in range(0, len(int_p), 2):
        if (int_p[i]) != 0:
            c = sqrt((int_p[i] - main_car.x+main_car.width/2) ** 2 + (int_p[i + 1] - main_car.y+main_car.height/2) ** 2)
        else:
            c = 500
        distance.append(int(c))
    if len(distance) > 194:
        pass
    return distance


# it makes zero point of intersaction with every side of a car
def line_rect_intersection(line, rect):
    def line_intersection(line1, line2):

        d = (line2[1][1] - line2[0][1]) * (line1[1][0] - line1[0][0]) - (line2[1][0] - line2[0][0]) * (
                line1[1][1] - line1[0][1])
        n_a = (line2[1][0] - line2[0][0]) * (line1[0][1] - line2[0][1]) - (line2[1][1] - line2[0][1]) * (
                line1[0][0] - line2[0][0])
        n_b = (line1[1][0] - line1[0][0]) * (line1[0][1] - line2[0][1]) - (line1[1][1] - line1[0][1]) * (
                line1[0][0] - line2[0][0])
        # Determinant
        if d == 0:
            return None

        ua = n_a / d
        ub = n_b / d

        if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
            x = line1[0][0] + (ua * (line1[1][0] - line1[0][0])) - 7.5
            y = line1[0][1] + (ua * (line1[1][1] - line1[0][1])) - 7.5
            return (int(x), int(y))
        else:
            return None

    # Dividing rect to lines
    # Dividing rect to lines
    lines = []
    lines.append(((rect.x, rect.y), (rect.x + rect.width, rect.y)))  # front line
    lines.append(((rect.x, rect.y), (rect.x, rect.y + rect.height)))  # left line
    lines.append(((rect.x + rect.width, rect.y), (rect.x + rect.width, rect.y + rect.height)))  # right line
    lines.append(((rect.x, rect.y + rect.height), (rect.x + rect.width, rect.y + rect.height)))  # bottom line

    points = []
    for i in range(len(lines)):
        points.append(line_intersection(lines[i], line))

    return points


def update_fps(clock):
    fps = str(int(clock.get_fps()))
    return fps


def draw_text(text, pos, color=BLACK, size=20, style='freesansbold.ttf'):
    FONT = pygame.font.Font(style, size)
    rend_text = FONT.render(text, True, color)
    win.blit(rend_text, pos)


def draw_road(road_y):
    win.blit(road, (0, road_y))
    win.blit(road, (0, road_y - WIN_HEIGHT))


def out_of_the_screen(cars):
    for i in range(len(cars) - 1):
        if cars[i].y >= 900 or cars[i].y <= -350:  # Out of the screen
            cars.pop(i)
            return


def cars_speed_control(cars):
    def speed_up(i, j):  # Direction == 0 => form up to down
        if cars[i].speed > cars[j].speed:
            cars[j].speed = cars[i].speed
        elif cars[i].speed < cars[j].speed:
            cars[i].speed = cars[j].speed

    for i in range(len(cars) - 1):
        for j in range(1, len(cars) - 1):
            if (i != j) and (cars[i].x == cars[j].x):
                if cars[i].speed != cars[j].speed:
                    if abs(cars[i].y + cars[i].height - cars[j].y) < 70 or abs(
                            cars[j].y + cars[j].height - cars[i].y) < 70:
                        speed_up(i, j)


def check_for_overlapying(cars):
    for i in range(len(cars) - 1):
        for j in range(1, len(cars) - 1):
            if j != i:
                if cars[i].x == cars[j].x:
                    if cars[i].rect.colliderect(cars[j].rect):
                        cars.pop(j)
                        return


def cars_append(cars):
    cars.append(Cars(random.randint(1, 8)))


# Button's functions
def hiding_sensors(button, car):
    button.clicked = not button.clicked
    car.hide_sensors = not car.hide_sensors


plot_score_x = np.array([0, ])
plot_score_y = np.array([0, ])
plot_loss_x = np.array([0, ])
plot_loss_y = np.array([0, ])


def make_loss_plot(generation, loss):
    global plot_loss_x, plot_loss_y
    plot_loss_x = np.append(plot_loss_x, generation)
    plot_loss_y = np.append(plot_loss_y, loss)
    plt.cla()
    plt.plot(plot_loss_x, plot_loss_y, color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('Mean Square Error, Quadratic loss')
    if len(plot_loss_x) >= 2:
        try:
            poly_reg = PolynomialFeatures(degree=3)
            x_poly = poly_reg.fit_transform(plot_loss_x.reshape(-1, 1))
            pol_reg = LinearRegression()
            pol_reg.fit(x_poly, plot_loss_y.reshape(-1, 1))

        except:
            poly_reg = PolynomialFeatures(degree=3)
            x_poly = poly_reg.fit_transform(plot_loss_x.reshape(-1, 1))
            pol_reg = LinearRegression()
            pol_reg.fit(x_poly, plot_loss_y.reshape(-1, 1))

        y_pred_poly = pol_reg.predict(poly_reg.fit_transform(plot_loss_x.reshape(-1, 1)))
        plt.tight_layout()
        plt.plot(plot_loss_x, y_pred_poly, '--g')

    plt.savefig('src/loss.png')


def make_score_plot(x, y):
    global plot_score_x, plot_score_y
    plot_score_x = np.append(plot_score_x, int(x))
    plot_score_y = np.append(plot_score_y, int(y))
    plt.cla()
    plt.plot(plot_score_x, plot_score_y)
    plt.title('''Generation's max score''')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    if len(plot_score_x) >= 2:
        try:
            poly_reg = PolynomialFeatures(degree=3)
            x_poly = poly_reg.fit_transform(plot_score_x.reshape(-1, 1))
            pol_reg = LinearRegression()
            pol_reg.fit(x_poly, plot_score_y.reshape(-1, 1))
        except:
            poly_reg = PolynomialFeatures(degree=3)
            x_poly = poly_reg.fit_transform(plot_score_x.reshape(-1, 1))
            pol_reg = LinearRegression()
            pol_reg.fit(x_poly, plot_score_y.reshape(-1, 1))

        y_pred_poly = pol_reg.predict(poly_reg.fit_transform(plot_score_x.reshape(-1, 1)))
        plt.tight_layout()
        plt.plot(plot_score_x, y_pred_poly, '--r')
    plt.savefig('src/plot.png')


class X10(Button):
    def __init__(self, rect):
        super().__init__(text='x10', rect=rect)

    def click(self):
        self.clicked = not self.clicked
        if self.clicked:
            pygame.time.set_timer(THINKING, 20)  # when 15 it works normal
            pygame.time.set_timer(CARS_SPAWN, 1500)
        else:
            pygame.time.set_timer(THINKING, 100)  # 100
            pygame.time.set_timer(CARS_SPAWN, 7500)  # 9000


class Settings(Button):
    def __init__(self, rect):
        super().__init__(text='Settings', rect=rect)

    def click(self):
        self.clicked = not self.clicked


class Control(object):
    def __init__(self, agent, rect=(GAME_WIDTH+160, 52, 100, 20), topleft=(GAME_WIDTH+15, 52), message='Learning rate:'):
        self.input = TextBox(rect, command=self.change_params,
                             clear_on_enter=True, inactive_on_enter=False, active=False)
        self.prompt = self.make_prompt(topleft=topleft, message=message)
        self.rect = rect
        self.done = False
        self.agent = agent
        pygame.key.set_repeat(200, 70)

    def make_prompt(self, topleft=(100, 100), message=''):
        font = FONT
        message = message
        rend = font.render(message, True, pygame.Color("black"))
        return (rend, rend.get_rect(topleft=topleft))

    def event_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            self.input.get_event(event)

    def change_params(self, gamma=0.99, lr=0.002, epsilon=1.0, max_mem_size=100000, eps_end=0.01, eps_dec=0.00015):
        global Gamma, Lr, Max_size, Epsilon, Eps_dec, Eps_end, params
        try:
            if gamma != 0.99 and gamma != None:
                Gamma = float(gamma)
                params = True
            else:
                Gamma = 0.99
            if lr != 0.001 and lr != None:
                Lr = float(lr)
                params = True
            else:
                Lr = 0.001
            if epsilon <= 1.0 and epsilon != None:
                Epsilon = float(epsilon)
                params = True
            else:
                Epsilon = 1.0
            if max_mem_size != 100000 and max_mem_size != None:
                Max_size = int(max_mem_size)
                params = True
            else:
                Max_size = 500000
            if eps_end != 0.01 and eps_end != None:
                Eps_end = float(eps_end)
                params = True
            else:
                Eps_end = 0.01
            if eps_dec != 0.0001 and eps_dec != None:
                Eps_dec = float(eps_dec)
                params = True
            else:
                Eps_dec = 0.0001

            self.input = TextBox(self.rect, command=self.change_params,
                                 clear_on_enter=True, inactive_on_enter=False, active=True,
                                 active_color=pygame.Color("green"))  # Need to make it iterable object


        except:
            print("Please input a valid value.")  # Делать обводку бокса красным светом, если все хорошо - зеленым
            self.input = TextBox(self.rect, command=self.change_params,
                                 clear_on_enter=True, inactive_on_enter=False, active=True,
                                 active_color=pygame.Color("red"))


from textbox import TextBox


def main():
    pygame.draw.rect(win, WHITE, (0, 0, WIN_WIDTH, WIN_HEIGHT))

    clock = pygame.time.Clock()
    generation = 0
    main_car = Car()
    sensors_but = Button(rect=(GAME_WIDTH + 35, 10, 100, 30), text='Sensors', clicked=True)
    save_but = Button(rect=(GAME_WIDTH + 100 + 50, 10, 130, 30), text='Save model')
    load_but = Button(rect=(GAME_WIDTH + 200 + 95, 10, 130, 30), text='Load model')
    x10_but = X10(rect=(GAME_WIDTH + 300 + 140, 10, 70, 30))
    settings_but = Settings((GAME_WIDTH + 400 + 125, 10, 110, 30))
    box_lr = Control(main_car.brain, (GAME_WIDTH + 170, 52, 100, 20), topleft=(GAME_WIDTH + 25, 52))
    box_epsilon = Control(main_car.brain, (GAME_WIDTH + 170, 84, 100, 20), topleft=(GAME_WIDTH + 25, 84),
                          message='Epsilon:')
    box_gamma = Control(main_car.brain, (GAME_WIDTH + 170, 116, 100, 20), topleft=(GAME_WIDTH + 25, 116),
                        message='Gamma:')
    box_eps_end = Control(main_car.brain, (GAME_WIDTH + 415, 84, 100, 20), topleft=(GAME_WIDTH + 300, 84),
                          message='Eps_end:')
    box_max_mem_size = Control(main_car.brain, (GAME_WIDTH + 415, 52, 100, 20), topleft=(GAME_WIDTH + 300, 52),
                               message='Mem_size:')
    box_eps_dec = Control(main_car.brain, (GAME_WIDTH +415, 116, 100, 20), topleft=(GAME_WIDTH + 300, 116),
                          message='Eps_dec:')
    # TODO: drawing minimal distance
    road_y = 0
    cars = [Cars(random.randint(1, 8))]
    prev_score = 0
    max_score = 0
    global plot_score_x, plot_score_y, plot_loss_x, plot_loss_y
    plt.figure(figsize=(15, 5), dpi=57)
    plt.plot(plot_score_x, plot_score_y)
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('''Generation's max score''')
    plt.tight_layout()
    plt.savefig('src/plot.png')

    plt.figure(figsize=(15, 5), dpi=57)
    plt.plot(generation, 0)
    plt.title('Mean Square Error, Quadratic loss')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('src/loss.png')
    # Main Loop
    while True:
        clock.tick()  # FPS: 40
        # Background on the right
        pygame.draw.rect(win, WHITE, (GAME_WIDTH, 0, WIN_WIDTH - GAME_WIDTH, WIN_HEIGHT))
        draw_road(road_y)
        if road_y >= WIN_HEIGHT:
            road_y = 0
        else:
            road_y += 2  # Speed of moving road

        def redraw_window():
            # Drawing cars
            for i in range(0, len(cars) - 1):
                win.blit(cars[i].img, (cars[i].x, cars[i].y))

            # Main_car
            main_car.draw_sensors(8)
            win.blit(main_car.img, (main_car.x, main_car.y))
            # Plot
            score_graph = pygame.image.load('src/plot.png')
            win.blit(score_graph, (GAME_WIDTH + 20, 138))

            loss_graph = pygame.image.load('src/loss.png')
            win.blit(loss_graph, (GAME_WIDTH + 20, 416))

            sensors_but.draw()
            save_but.draw()
            load_but.draw()
            x10_but.draw()
            settings_but.draw()
            # FPS
            # draw_text('FPS: ' + update_fps(clock), (WIN_WIDTH - 85, 18))
            # Amount of cars
            draw_text('Cars on the road: ' + f'{len(cars)}', (GAME_WIDTH + 20, 52))
            # draw_text('Main_cars alive: ' + f'{len(s)}', (GAME_WIDTH + 170, 108))
            draw_text('Generation: ' + f'{generation}', (GAME_WIDTH + 280, 52))
            # Car brain's data
            draw_text('Score: ' + f'{int(main_car.score)}', (GAME_WIDTH + 400, 116))
            draw_text('Previous score: ' + f'{int(prev_score)}', (GAME_WIDTH + 600, 116))
            draw_text('Max score: ' + f'{int(max_score)}', (GAME_WIDTH + 160, 116))
            draw_text('Action type: ' + f'{main_car.action_type}', (GAME_WIDTH + 700, 52))
            draw_text(f'Epsilon: {"%.7f" % main_car.epsilon}', (GAME_WIDTH + 480, 52))
            draw_text('Up: ' + f'{"%.3f" % main_car.actions[0]}', (GAME_WIDTH + 20, 84))
            draw_text('Down: ' + f'{"%.3f" % main_car.actions[1]}', (GAME_WIDTH + 170, 84))
            draw_text('Left: ' + f'{"%.3f" % main_car.actions[2]}', (GAME_WIDTH + 360, 84))
            draw_text('Right: ' + f'{"%.3f" % main_car.actions[3]}', (GAME_WIDTH + 520, 84))
            draw_text('Nothing: ' + f'{"%.3f" % main_car.actions[4]}', (GAME_WIDTH + 700, 84))

        redraw_window()

        if settings_but.clicked == True:
            pygame.draw.rect(win, WHITE, (GAME_WIDTH, 50, WIN_WIDTH - GAME_WIDTH, 90))
            box_lr.input.update()
            box_lr.input.draw(win)
            box_epsilon.input.update()
            box_epsilon.input.draw(win)
            box_gamma.input.update()
            box_gamma.input.draw(win)
            box_eps_end.input.update()
            box_eps_end.input.draw(win)
            box_max_mem_size.input.update()
            box_max_mem_size.input.draw(win)
            box_eps_dec.input.update()
            box_eps_dec.input.draw(win)
            win.blit(*box_eps_dec.prompt)
            win.blit(*box_max_mem_size.prompt)
            win.blit(*box_eps_end.prompt)
            win.blit(*box_gamma.prompt)
            win.blit(*box_epsilon.prompt)
            win.blit(*box_lr.prompt)
            draw_text('Standard value:', (GAME_WIDTH + 630, 45), size=18)
            draw_text('Learning rate: 0.001', (GAME_WIDTH + 545, 72), size=14)
            draw_text('Gamma: 0.99', (GAME_WIDTH + 545, 92), size=14)
            draw_text('Max_mem: 300,000', (GAME_WIDTH + 545, 112), size=14)
            draw_text('Epsilon: 1.0', (GAME_WIDTH + 745, 112), size=14)
            draw_text('Eps_end: 0.01', (GAME_WIDTH + 745, 92), size=14)
            draw_text('Eps_dec: 15e-5', (GAME_WIDTH + 745, 72), size=14)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if settings_but.clicked == True:
                box_lr.input.get_event(event)
            if settings_but.clicked == True:
                box_gamma.input.get_event(event)
            if settings_but.clicked == True:
                box_epsilon.input.get_event(event)
            if settings_but.clicked == True:
                box_eps_end.input.get_event(event)
            if settings_but.clicked == True:
                box_max_mem_size.input.get_event(event)
            if settings_but.clicked == True:
                box_eps_dec.input.get_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                # 1 is the left mouse button, 2 is middle, 3 is right.
                if event.button == 1:
                    # "event.pos" is the mouse position. If mouse pos == button pos -> click
                    if sensors_but.rect.collidepoint(event.pos):
                        hiding_sensors(sensors_but, main_car)
                    if x10_but.rect.collidepoint(event.pos):
                        x10_but.click()
                    if settings_but.rect.collidepoint(event.pos):
                        settings_but.click()
                    if save_but.rect.collidepoint(event.pos):
                        save_model(main_car.brain)
                    if load_but.rect.collidepoint(event.pos):
                        main_car.brain = load_model(lr=Lr, n_actions=5, input_dims=[INPUT_DIMS], gamma=Gamma,
                                                    epsilon=Epsilon)
                        main_car = Car(main_car.brain)
                        if sensors_but.clicked == False:
                            main_car.hide_sensors = False

                        generation = 0
                        prev_score = 0
                        max_score = 0

                        plot_score_x = np.array([0, ])
                        plot_score_y = np.array([0, ])
                        plot_loss_x = np.array([0, ])
                        plot_loss_y = np.array([0, ])

                        plt.figure(figsize=(15, 5), dpi=57)
                        plt.cla()
                        plt.plot(plot_loss_x, plot_loss_y, color='orange')
                        plt.xlabel('Generation')
                        plt.ylabel('Loss')
                        plt.title('Mean Square Error, Quadratic loss')
                        plt.tight_layout()
                        plt.savefig('src/loss.png')

                        plt.figure(figsize=(15, 5), dpi=57)
                        plt.cla()
                        plt.plot(plot_score_x, plot_score_y)
                        plt.title('''Generation's max score''')
                        plt.xlabel('Generation')
                        plt.ylabel('Score')
                        plt.tight_layout()
                        plt.savefig('src/plot.png')

                        cars = [Cars(random.randint(1, 8))]
            else:
                # Side movements animation
                if event.type == RESET_SET_PARAMS:
                    box_eps_dec.input = TextBox(box_eps_dec.rect, command=box_eps_dec.change_params,
                                                clear_on_enter=True, inactive_on_enter=False, active=False,
                                                active_color=pygame.Color("blue"))
                    box_max_mem_size.input = TextBox(box_max_mem_size.rect, command=box_max_mem_size.change_params,
                                                     clear_on_enter=True, inactive_on_enter=False, active=False,
                                                     active_color=pygame.Color("blue"))
                    box_eps_end.input = TextBox(box_eps_end.rect, command=box_eps_end.change_params,
                                                clear_on_enter=True, inactive_on_enter=False, active=False,
                                                active_color=pygame.Color("blue"))
                    box_gamma.input = TextBox(box_gamma.rect, command=box_gamma.change_params,
                                              clear_on_enter=True, inactive_on_enter=False, active=False,
                                              active_color=pygame.Color("blue"))
                    box_epsilon.input = TextBox(box_epsilon.rect, command=box_epsilon.change_params,
                                                clear_on_enter=True, inactive_on_enter=False, active=False,
                                                active_color=pygame.Color("blue"))
                    box_lr.input = TextBox(box_lr.rect, command=box_lr.change_params,
                                           clear_on_enter=True, inactive_on_enter=False, active=False,
                                           active_color=pygame.Color("blue"))
                if event.type == THINKING:
                    for i in range(0, len(cars) - 1):
                        cars[i].move()
                    main_car.move_right_left(main_car)
                    main_car.move_up_down()
                    collision = main_car.collide(cars)
                    if collision == 1:
                        main_car.reward = -10
                        main_car.think(cars, 0)
                        make_score_plot(generation, main_car.score)
                        make_loss_plot(generation, main_car.loss)
                        prev_score = main_car.score
                        if prev_score > max_score:
                            max_score = prev_score

                        main_car = Car(main_car.brain)
                        if sensors_but.clicked == False:
                            main_car.hide_sensors = False
                        generation += 1
                        cars = [Cars(random.randint(1, 8))]

                    main_car.think(cars)
                    main_car.brain.learn()
                    # main_car.brain.learn(main_car.brain.memory)
                    main_car.score += 1 / 2

                    check_for_overlapying(cars)
                    cars_speed_control(cars)
                    out_of_the_screen(cars)
                if event.type == CARS_SPAWN:
                    cars_append(cars)
                if event.type == DRAIWING:
                    # Moving road
                    pass


if __name__ == '__main__':
    main()
