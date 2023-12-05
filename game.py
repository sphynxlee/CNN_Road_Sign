import pygame
import time
import random

# Initialize pygame
pygame.init()

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
car_image = pygame.image.load("assets\Audi_Skinny_2.png")
# background_image = NULL
intro_background_image = pygame.image.load(r"assets\intro_background.jpg")
paused_background_image = pygame.image.load(r"assets\pause_icon.png")
gameplay_background_image = pygame.image.load(r"assets\background_image.jpg")
road_image = pygame.image.load(r"assets\road_image.jpg")
stop_sign_image = pygame.image.load("assets\stop_sign.jpg")
crosswalk_image = pygame.image.load("assets\crosswalk_sign.png")
speedlimit80_sign_image = pygame.image.load("assets\speedlimit_sign.png")
trafficlight_sign_image = pygame.image.load(r"assets\traffic_light_sign.png")
crossing_image = pygame.image.load("assets\crossing_image.png")

trafficlight_green_image = pygame.image.load(r'assets\trafficlight_green.png')
trafficlight_yellow_image = pygame.image.load(r'assets\trafficlight_yellow.png')
trafficlight_red_image = pygame.image.load(r'assets\trafficlight_red.png')


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

def show_sign(x_sign_start, y_sign_start, sign_type):
    if sign_type == 'stop':
        sign_picture = stop_sign_image
    if sign_type == 'crosswalk':
        sign_picture = crosswalk_image
    if sign_type == 'trafficlight':
        sign_picture = trafficlight_sign_image
    if sign_type == 'speedlimit80':
        sign_picture = speedlimit80_sign_image
    
    gamedisplay.blit(sign_picture, (x_sign_start, y_sign_start))

def score_system(passed, score):
    font = pygame.font.SysFont(None, 25)

    score = font.render("Score "+str(score), True, black)

    gamedisplay.blit(score, (0, 30))

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
    font = pygame.font.SysFont(None, 25)
    x = (display_width*0.35)
    y = (display_height*0.95)

    gamedisplay.blit(gameplay_background_image, (0,0))

    gamedisplay.blit(road_image, (275,0)) 

    gamedisplay.blit(car_image, (x, y))

    score = font.render("SCORE: 0", True, black)

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
    global paused
    x = (display_width*0.35)
    y = (display_height*0.95)
    x_change = 0
    obstacle_speed = 9
    # sign = 'trafficlight'
    sign = random.choice(['stop', 'crosswalk', 'trafficlight', 'speedlimit80'])
    y_change = -2
    # obs_startx = 700
    x_sign_start = 560
    y_sign_start = 200
    # obs_starty = 400

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

        # Update car
        if sign == "crosswalk":
            gamedisplay.blit(crossing_image, (275, 400))
            y += y_change
            car(x, y)
            if y == 478:
                y_change = 0
                time.sleep(1)
                y_change = -4
        
        elif sign == "speedlimit80":
            y_change = -8
            y += y_change
            car(x,y)

        elif sign == "stop":
            y += y_change
            car(x, y)
            if y == 478:
                y_change = 0
                time.sleep(2)
                y_change = -4

        elif sign == "trafficlight":
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
            game_loop()

        button("Pause", 650, 0, 150, 50, blue, bright_blue, "pause")
        pygame.display.update()
        clock.tick(60)



main_menu()
game_loop()
pygame.quit()
quit()





