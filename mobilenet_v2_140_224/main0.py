from random import random as rand
from random import randint

import numpy as np
import operator
import pygame
import math

class Particle:
    particles = 0

    def __init__(self):
        self.r = rand() * 0.01
        self.m = math.pi * self.r ** 2

        self.x = rand() * (1 - self.r * 2) + self.r
        self.y = rand() * (1 - self.r * 2) + self.r
        self.z = rand() * (1 - self.r * 2) + self.r

        self.vx = rand() * 2 - 1
        self.vy = rand() * 2 - 1
        self.vz = rand() * 2 - 1

        self.ax = 0
        self.ay = 0 # 9.80665
        self.az = 0

        self.i = Particle.particles
        Particle.particles += 1

        self.colliding = False

    @property
    def momentum(self):
        return self.m * self.v

    def distance(self, particle):
        dx = self.x - particle.x
        dy = self.y - particle.y
        dz = self.z - particle.z

        dx *= dx
        dy *= dy
        dz *= dz

        return math.sqrt(dx + dy + dz)

    def update(self, dt):
        x = self.x
        y = self.y
        z = self.z

        x += self.vx * dt + self.ax * dt * dt * 0.5
        y += self.vy * dt + self.ay * dt * dt * 0.5
        z += self.vz * dt + self.az * dt * dt * 0.5

        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt

        """
        if self.colliding:
            if self.r < x < 1 - self.r and self.r < y < 1 - self.r and self.r < z < 1 - self.r:
                self.colliding = False
        else:
            if x < self.r or x > 1 - self.r:
                self.vx *= -0.8
                self.colliding = True

            if y < self.r or y > 1 - self.r:
                self.vy *= -0.8
                self.colliding = True

            if z < self.r or z > 1 - self.r:
                self.vz *= -0.8
                self.colliding = True
        """

        damping = 0.8

        if x < self.r:
            x = self.r
            if self.vx < 0:
                self.vx *= damping
            self.vx = abs(self.vx)
            self.colliding = True

        if x > 1 - self.r:
            x = 1 - self.r
            if self.vx > 0:
                self.vx *= damping
            self.vx = -abs(self.vx)
            self.colliding = True

        if y < self.r:
            y = self.r
            if self.vy < 0:
                self.vy *= damping
            self.vy = abs(self.vy)
            self.colliding = True

        if y > 1 - self.r:
            y = 1 - self.r
            if self.vy > 0:
                self.vy *= damping
            self.vy = -abs(self.vy)
            self.colliding = True

        if z < self.r:
            z = self.r
            if self.vz < 0:
                self.vz *= damping
            self.vz = abs(self.vz)
            self.colliding = True

        if z > 1 - self.r:
            z = 1 - self.r
            if self.vz > 0:
                self.vz *= damping
            self.vz = -abs(self.vz)
            self.colliding = True

        # x %= 1
        # y %= 1
        # z %= 1

        # self.vx -= self.vx % 0.01
        # self.vy -= self.vy % 0.01
        # self.vz -= self.vz % 0.01

        # self.vx = round(self.vx, 1)
        # self.vy = round(self.vy, 1)
        # self.vz = round(self.vz, 1)

        # print(self.vx, self.vy, self.vz)

        return x, y, z

class Map:
    def __init__(self, precision):
        self.precision = precision
        self.particles = {}
        self.map = {}

    def add(self, particle: Particle):
        x = particle.x
        y = particle.y
        z = particle.z

        # x = round(x, self.precision)
        # y = round(y, self.precision)
        # z = round(z, self.precision)

        x -= x % self.precision
        y -= y % self.precision
        z -= z % self.precision

        c = (x, y, z)

        if c in self.map:
            self.map[c].add(particle.i)
        else:
            self.map[c] = set([particle.i])

        self.particles[particle.i] = particle

    def move(self, particle: Particle, xn: float, yn: float, zn: float):
        x = particle.x
        y = particle.y
        z = particle.z

        # x = round(x, self.precision)
        # y = round(y, self.precision)
        # z = round(z, self.precision)

        x -= x % self.precision
        y -= y % self.precision
        z -= z % self.precision

        c = (x, y, z)

        self.map[c].remove(particle.i)

        if not self.map[c]:
            del self.map[c]

        particle.x = xn
        particle.y = yn
        particle.z = zn

        xn -= xn % self.precision
        yn -= yn % self.precision
        zn -= zn % self.precision

        c = (xn, yn, zn)

        if c in self.map:
            self.map[c].add(particle.i)
        else:
            self.map[c] = set([particle.i])

    def update(self, particle: Particle, *, dt: float):
        x, y, z = particle.update(dt)
        self.move(particle, x, y, z)
        particle.x = x
        particle.y = y
        particle.z = z

    def collide(self, p1, p2, cor=1):
        vx1 = (p1.m * p1.vx + p2.m * p2.vx + p2.m * cor * (p2.vx - p1.vx)) / (p1.m + p2.m)
        vy1 = (p1.m * p1.vy + p2.m * p2.vy + p2.m * cor * (p2.vy - p1.vy)) / (p1.m + p2.m)
        vz1 = (p1.m * p1.vz + p2.m * p2.vz + p2.m * cor * (p2.vz - p1.vz)) / (p1.m + p2.m)

        vx2 = (p1.m * p1.vx + p2.m * p2.vx + p1.m * cor * (p1.vx - p2.vx)) / (p1.m + p2.m)
        vy2 = (p1.m * p1.vy + p2.m * p2.vy + p1.m * cor * (p1.vy - p2.vy)) / (p1.m + p2.m)
        vz2 = (p1.m * p1.vz + p2.m * p2.vz + p1.m * cor * (p1.vz - p2.vz)) / (p1.m + p2.m)

        p1.vx = vx1
        p1.vy = vy1
        p1.vz = vz1

        p2.vx = vx2
        p2.vy = vy2
        p2.vz = vz2

class Scene:
    g = 9.80665
    freq = 60

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 50, 50)
    GREEN = (50, 255, 50)

    def __init__(self, n=100):
        self.particles = [Particle() for i in range(n)]
        self.collision = [[False for _ in range(n)] for _ in range(n)]

        self.cor = [[0.93 for _ in range(n)] for _ in range(n)]
        self.cof = [[0.2 for _ in range(n)] for _ in range(n)]

        self.map = Map(0.1)

        for particle in self.particles:
            self.map.add(particle)

    def begin(self):
        pygame.init()
        screen = pygame.display.set_mode((500, 500))
        pygame.display.set_caption('Particle Collisions')
        fps = pygame.time.Clock()
        self.t = 0
        self.dt = 1 / self.freq
        return screen, fps

    def handle_collision(self, ij):
        assert False
        i, j = ij
        pi, pj = self.particles[i], self.particles[j]
        if pi.distance(pj) < pi.r + pj.r:
            if not self.collision[i][j]:
                return i, j
                self.collision[i][j] = True
        else:
            self.collision[i][j] = False

    def update(self):
        self.t += self.dt

        for particle in self.particles:
            self.map.update(particle, dt = self.dt)

        for key, particles in self.map.map.items():
            if len(particles) > 1:
                particles = list(particles)
                collisions = []

                for i in range(len(particles)):
                    pi = particles[i]
                    pi = self.particles[pi]
                    for j in range(i + 1, len(particles)):
                        pj = particles[j]
                        pj = self.particles[pj]

                        if pi.distance(pj) < pi.r + pj.r:
                            if not self.collision[pi.i][pj.i]:
                                collisions.append((pi.i, pj.i))
                                self.collision[pi.i][pj.i] = True
                        elif self.collision[pi.i][pj.i]:
                            self.collision[pi.i][pj.i] = False

                for i, j in collisions:
                    p1, p2 = self.particles[i], self.particles[j]
                    self.map.collide(p1, p2, cor = self.cor[i][j])

    def render(self, screen, fps):
        xc = 0.5
        yc = 0.2
        zc = 0.5
        screen.fill(self.BLACK)
        for particle in self.particles:
            # X' = ((X - Xc) * (F/Z)) + Xc
            # Y' = Y / Z * (Z - Zc)

            dist = math.sqrt((particle.x - 0.5) ** 2 + (particle.y - 0.5) ** 2 + (particle.z - 0.5) ** 2)
            color = int(255 * (1 - dist))
            color = (color, color, color)
            # x = (particle.x - xc) * (particle.z - zc) / particle.z + xc
            # y = (particle.y - yc) * (particle.z - zc) / particle.z + yc
            x = particle.x
            y = particle.y
            pygame.draw.circle(screen, color, (x * 500, y * 500), particle.r * 500 / dist, 0)
        # x, y, z = self.center_of_mass
        # pygame.draw.circle(screen, self.RED, (x * 500, y * 500), 1 / math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2), 0)
        pygame.display.update()
        fps.tick(self.freq)

    def run(self):
        screen, fps = self.begin()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            self.update()
            self.render(screen, fps)

    @property
    def center_of_mass(self):
        m = sum([particle.m for particle in self.particles])
        x = sum([particle.x * particle.m for particle in self.particles]) / m
        y = sum([particle.y * particle.m for particle in self.particles]) / m
        z = sum([particle.z * particle.m for particle in self.particles]) / m
        return x, y, z

    @property
    def energy(self):
        K = 0
        for particle in self.particles:
            K += 0.5 * particle.m * particle.v ** 2
        return K

    @property
    def momentum(self):
        return sum([particle.momentum for particle in self.particles])

if __name__ == '__main__':
    scene = Scene(n = 1000)
    scene.run()
