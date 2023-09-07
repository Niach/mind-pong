import pygame


class PongGame:
    WIDTH = 1920
    HEIGHT = 1080

    @classmethod
    def draw_arrow_left(cls, screen):
        pygame.draw.polygon(screen, "white",
                            [(cls.WIDTH * 0.5 - cls.WIDTH * 0.039, cls.HEIGHT * 0.139),
                             (cls.WIDTH * 0.5 - cls.WIDTH * 0.039, cls.HEIGHT * 0.278),
                             (cls.WIDTH * 0.5 - cls.WIDTH * 0.117, cls.HEIGHT * 0.208)])
        pygame.draw.rect(screen, "white",
                         pygame.Rect(cls.WIDTH * 0.5 - cls.WIDTH * 0.039, cls.HEIGHT * 0.187, cls.WIDTH * 0.078,
                                     cls.HEIGHT * 0.042))

    @classmethod
    def draw_arrow_right(cls, screen):
        pygame.draw.polygon(screen, "white",
                            [(cls.WIDTH * 0.5 + cls.WIDTH * 0.039, cls.HEIGHT * 0.139),
                             (cls.WIDTH * 0.5 + cls.WIDTH * 0.039, cls.HEIGHT * 0.278),
                             (cls.WIDTH * 0.5 + cls.WIDTH * 0.117, cls.HEIGHT * 0.208)])
        pygame.draw.rect(screen, "white",
                         pygame.Rect(cls.WIDTH * 0.5 - cls.WIDTH * 0.039, cls.HEIGHT * 0.187, cls.WIDTH * 0.078,
                                     cls.HEIGHT * 0.042))

    @classmethod
    def draw_rest(cls, screen):
        pygame.draw.circle(screen, "white", (cls.WIDTH * 0.5, cls.HEIGHT * 0.208), cls.WIDTH * 0.039)
        pygame.draw.circle(screen, "black", (cls.WIDTH * 0.5, cls.HEIGHT * 0.208), cls.WIDTH * 0.023)

    def __init__(self, left_event, right_event, rest_event):
        self.left_event = left_event
        self.right_event = right_event
        self.rest_event = rest_event
        pygame.init()
        self.screen = pygame.display.set_mode((1920, 1080))
        self.clock = pygame.time.Clock()
        self.running = True

        self.current_direction = "REST"
        self.current_x = PongGame.WIDTH * 0.5 - PongGame.WIDTH * 0.039

    def run(self):
        while self.running:
            self.update()
        self.quit()

    def update(self):
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # fill the screen with a color to wipe away anything from last frame

        self.screen.fill("black")

        # RENDER YOUR GAME HERE
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            self.running = False

        if self.left_event.is_set():
            self.current_direction = "LEFT"
            self.current_x = PongGame.WIDTH * 0.805
            self.left_event.clear()

        if self.right_event.is_set():
            self.current_direction = "RIGHT"
            self.current_x = PongGame.WIDTH * 0.117
            self.right_event.clear()

        if self.rest_event.is_set():
            self.current_direction = "REST"
            self.current_x = PongGame.WIDTH * 0.5 - PongGame.WIDTH * 0.039
            self.rest_event.clear()

        if self.current_direction == "LEFT":
            self.current_x = self.current_x - 0.5
            PongGame.draw_arrow_left(self.screen)

        elif self.current_direction == "RIGHT":
            self.current_x = self.current_x + 0.5
            PongGame.draw_arrow_right(self.screen)

        elif self.current_direction == "REST":
            PongGame.draw_rest(self.screen)

        pygame.draw.rect(self.screen, "white",
                         pygame.Rect(self.current_x, PongGame.HEIGHT * 0.833, PongGame.WIDTH * 0.078, PongGame.HEIGHT * 0.02))

        # update the display
        pygame.display.flip()
        self.clock.tick(144)

    def quit(self):
        pygame.quit()