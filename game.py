import pygame
import time
import random
import os
from random_image import predict_image

# Initialize pygame
pygame.init()
pwd = os.getcwd()
score = 0

# Define colors here
gray = (119, 118, 110)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 200, 0)
blue = (0, 0, 200)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)
bright_blue = (0, 0, 255)

# Display dimensions
display_width = 1200
display_height = 800

# Setup a game display
gamedisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.update()
pygame.display.set_caption("CNN Car Game")
clock = pygame.time.Clock()

# Load assets
# this will load from current working directory
car_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\Audi_Skinny_2.png")
# background_image = NULL
intro_background_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\intro_background.jpg")
paused_background_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\pause_icon.png")
gameplay_background_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\background_image.jpg")
road_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\road_image.jpg")
stop_sign_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\stop_sign.jpg")
crosswalk_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\crosswalk_sign.png")
speedlimit80_sign_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\speedlimit_sign.png")
trafficlight_sign_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\traffic_light_sign.png")
crossing_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\crossing_image.png")

trafficlight_green_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\trafficlight_green.png")
trafficlight_yellow_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\trafficlight_yellow.png")
trafficlight_red_image = pygame.image.load(fr"{pwd}\CNN_Road_Sign\assets\trafficlight_red.png")
# Load assets
# this will load from current working directory
# car_image = pygame.image.load(fr"{pwd}\assets\Audi_Skinny_2.png")
# # background_image = NULL
# intro_background_image = pygame.image.load(fr"{pwd}\assets\intro_background.jpg")
# paused_background_image = pygame.image.load(fr"{pwd}\assets\pause_icon.png")
# gameplay_background_image = pygame.image.load(fr"{pwd}\assets\background_image.jpg")
# road_image = pygame.image.load(fr"{pwd}\assets\road_image.jpg")
# stop_sign_image = pygame.image.load(fr"{pwd}\assets\stop_sign.jpg")
# crosswalk_image = pygame.image.load(fr"{pwd}\assets\crosswalk_sign.png")
# speedlimit80_sign_image = pygame.image.load(fr"{pwd}\assets\speedlimit_sign.png")
# trafficlight_sign_image = pygame.image.load(fr"{pwd}\assets\traffic_light_sign.png")
# crossing_image = pygame.image.load(fr"{pwd}\assets\crossing_image.png")

# trafficlight_green_image = pygame.image.load(fr"{pwd}\assets\trafficlight_green.png")
# trafficlight_yellow_image = pygame.image.load(fr"{pwd}\assets\trafficlight_yellow.png")
# trafficlight_red_image = pygame.image.load(fr"{pwd}\assets\trafficlight_red.png")


# Initialize pause state
paused = False

end_of_road = False

# Def Text Object
def text_objects(text, font):
    textsurface = font.render(text, True, black)
    return textsurface, textsurface.get_rect()

# Def Button Object
def button(button_text, x, y, width, height, inactive_color, active_color, action=None):
    # Get current mouse pos
    mouse_pos = pygame.mouse.get_pos()
    # Get the current state of mouse buttons
    click = pygame.mouse.get_pressed()

    # Check if mouse is within the button's boundaries
    if x+width > mouse_pos[0] > x and y+height > mouse_pos[1] > y:
        # Draw button with active color
        pygame.draw.rect(gamedisplay, active_color, (x, y, width, height))

        # Check if left mouse is clicked and action is specified
        if click[0] == 1 and action != None:
            if action == "play":
                countdown()
            elif action == "quit":
                pygame.quit()
                quit()
                sys.exit()
            elif action == "menu":
                main_menu()
            elif action == "pause":
                pause()
            elif action == "unpause":
                unpause()
    else:
        # Draw inactive color
        pygame.draw.rect(gamedisplay, inactive_color, (x, y, width, height))

    smalltext = pygame.font.Font("freesansbold.ttf", 20)
    textsurf, textrect = text_objects(button_text, smalltext)
    textrect.center = ((x+(width/2)),(y+(height/2)))
    gamedisplay.blit(textsurf, textrect)

def pause():
    global paused

    # loop to handle events during pause state
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                sys.exit()
        gamedisplay.blit(paused_background_image, (0,0))
        largetext = pygame.font.Font('freesansbold.ttf', 115)
        TextSurf, TextRect = text_objects("PAUSED", largetext)
        TextRect.center = ((display_width/2),(display_height/2))
        gamedisplay.blit(TextSurf, TextRect)

        #Create Buttons
        button("Continue", 150, 450, 150, 50, green, bright_green, "unpause")

        button("Restart", 350, 450, 150, 50, blue, bright_blue, "play")

        button("Main Menu", 550, 450, 200, 50, red, bright_red, "menu")

        pygame.display.update()
        clock.tick(30)


def unpause():
    global paused
    paused = False

def show_sign(x_sign_start, y_sign_start, sign):
    # sign_picture = sign
    font = pygame.font.Font("freesansbold.ttf", 60)
    prediction = font.render(f"Prediction", True, black)

    gamedisplay.blit(prediction, (x_sign_start, y_sign_start-100))
    gamedisplay.blit(sign, (x_sign_start, y_sign_start))

def sign_action(x_sign_start, y_sign_start, sign_type):
    if sign_type == 'stop':
        sign_picture = stop_sign_image
    if sign_type == 'crosswalk':
        sign_picture = crosswalk_image
    if sign_type == 'trafficlight':
        sign_picture = trafficlight_sign_image
    if sign_type == 'speedlimit80':
        sign_picture = speedlimit80_sign_image
    

    gamedisplay.blit(sign_picture, (x_sign_start, y_sign_start))

def score_system(score):
    font = pygame.font.SysFont(None, 50)

    # score = font.render("Score: "+str(score), True, black)

    # gamedisplay.blit(score, (0, 50))

    score = font.render(f"Score: {score}", True, black)

    gamedisplay.blit(score, (0, 50))

def gameover_display(text):
    largetext = pygame.font.Font("freesansbold.ttf", 80)
    textsurf, textrect = text_objects(text, largetext)
    textrect.center = ((display_width/2), (display_height/2))

    gamedisplay.blit(textsurf, textrect)
    pygame.display.update()
    time.sleep(0.5)
    game_loop()

def loop():
    gameover_display("Loading New Loop...")

def background():
    gamedisplay.blit(gameplay_background_image (0, 0))

    gamedisplay.blit(road_image, (275,0))

def car(x, y):
    gamedisplay.blit(car_image, (x, y))

def select_random_file():
    pwd = os.getcwd()
    sign_folder_list = os.listdir(fr"{pwd}\CNN_Road_Sign\road_signs_img")
    # sign_folder_list = os.listdir(fr"{pwd}\road_signs_img")

    image_paths = []
    folder_choice = random.choice(['0 TrafficLight', '1 Stop', '2 speedlimit', '3 Crosswalk'])
    for folder in sign_folder_list:
        if folder == folder_choice:
            # print(fr"{pwd}\road_signs_img\{folder}")
            print(folder)

            choice = random.choice(os.listdir(fr"{pwd}\CNN_Road_Sign\road_signs_img\{folder}"))
            # choice = random.choice(os.listdir(fr"{pwd}\road_signs_img\{folder}"))
            print(choice)
            # image_paths.append()
    print(choice)
    print(fr"{pwd}\CNN_Road_Sign\road_signs_img\{folder_choice}\{choice}")
    return (fr"{pwd}\CNN_Road_Sign\road_signs_img\{folder_choice}\{choice}", folder_choice)
    # return (fr"{pwd}\road_signs_img\{folder_choice}\{choice}", folder_choice)

    # return random.choice([image_paths])


# Main Menu
def main_menu():
    main_menu = True
    while main_menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                sys.exit()

        # Intro Background
        gamedisplay.blit(intro_background_image, (0,0))

        # Render and display "CNN Car Game"
        largetext = pygame.font.Font('freesansbold.ttf',115)
        TextSurf, TextRect = text_objects("CNN Car Game", largetext)
        TextRect.center = (400, 100)
        gamedisplay.blit(TextSurf, TextRect)

        # Render and display Start and Quit buttons
        button("Start", 150, 520, 100, 50, green, bright_green, "play")
        button("Quit", 550, 520, 100, 50, red, bright_red, "quit")


        pygame.display.update()
        clock.tick(50)


def set_scene():
    font = pygame.font.SysFont(None, 50)
    x = (display_width*0.35)
    y = (display_height*0.95)

    gamedisplay.blit(gameplay_background_image, (0,0))

    gamedisplay.blit(road_image, (275,0))

    gamedisplay.blit(car_image, (x, y))

    score = font.render(f"Score: {0}", True, black)

    gamedisplay.blit(score, (0, 50))

    button("Pause", 650, 0, 150, 50, blue, bright_blue, "pause")

def countdown():
    countdown = True
    while countdown:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                sys.exit()

        # fill display
        gamedisplay.fill(gray)
        #prep game
        set_scene()

        largetext = pygame.font.Font('freesansbold.ttf', 115)


        # Display "GO!" in large font at the center of the screen
        TextSurf, TextRect = text_objects("GO!", largetext)
        TextRect.center = ((display_width/2), (display_height/2))
        gamedisplay.blit(TextSurf, TextRect)
        pygame.display.update()
        clock.tick(1)  # Delay for 1 second

        # Call the game loop function after the countdown is complete
        game_loop()

def game_loop():
    global paused, score
    x = (display_width*0.35)
    y = (display_height*0.95)
    x_change = 0
    obstacle_speed = 9
    # sign = 'trafficlight'

    sign_path, sign_folder_choice = select_random_file()
    print("SIGN PATH",sign_path)
    print("FOLDER CHOICE", sign_folder_choice)
    sign = pygame.image.load(sign_path)
    sign_prediction = predict_image(sign_path)
    print("PREDICTION", sign_prediction)
    print("LABEL", sign_folder_choice)
    # sign = random.choice(['stop', 'crosswalk', 'trafficlight', 'speedlimit80'])
    y_change = -2
    # obs_startx = 700
    x_sign_start = 560
    y_sign_start = 200
    # obs_starty = 400
    scored = False

    gameover = False
    while not gameover:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         x_change = -5

        paused = True
        gamedisplay.fill(gray)

        # Update game display
        gamedisplay.blit(gameplay_background_image, (0,0))

        gamedisplay.blit(road_image, (275,0))


        show_sign(x_sign_start, y_sign_start, sign)

        # Always show score
        score_system(score)

        # Update car
        if sign_prediction == 3:
            if sign_folder_choice == "3 Crosswalk" and scored == False:
                score += 1
                scored = True
            gamedisplay.blit(crossing_image, (275, 400))
            y += y_change
            car(x, y)
            if y == 478:
                y_change = 0
                time.sleep(1)
                y_change = -4

        elif sign_prediction == 2:
            if sign_folder_choice == "2 speedlimit" and scored == False:
                score += 1
                scored = True
            y_change = -8
            y += y_change
            car(x,y)

        elif sign_prediction == 1:
            if sign_folder_choice == "1 Stop" and scored == False:
                score += 1
                scored = True
            
            y += y_change
            car(x, y)
            if y == 478:
                y_change = 0
                time.sleep(2)
                y_change = -4

        elif sign_prediction == 0:
            if sign_folder_choice == "0 TrafficLight" and scored == False:
                score += 1
                scored = True
            y += y_change
            car(x, y)

            if y > 550:
                gamedisplay.blit(trafficlight_yellow_image, (250, 400))

            if y > 478 and 550 > y:
                gamedisplay.blit(trafficlight_red_image, (250, 400))

            if y == 478:
                y_change = 0
                time.sleep(3)
                y += -1
                gamedisplay.blit(trafficlight_green_image, (250, 400))
                time.sleep(0.3)

            if y < 478:
                gamedisplay.blit(trafficlight_green_image, (250, 400))
                y_change = -4
        else:
            x += x_change
            y += -2
            car(x, y)

        if y < 0:
            scored = False
            game_loop()

        button("Pause", 650, 0, 150, 50, blue, bright_blue, "pause")
        pygame.display.update()
        clock.tick(60)

# select_random_file()
main_menu()
game_loop()
pygame.quit()
quit()





