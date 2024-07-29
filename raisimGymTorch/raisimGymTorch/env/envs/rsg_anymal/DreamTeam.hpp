#pragma once

#include "../../RaisimGymEnv.hpp"
#include <random>

class DreamTeam {
public:
  explicit DreamTeam(float corridor_length, float corridor_width,
                     float min_obstacle_height, float max_obstacle_height,
                     float min_obstacle_length, float max_obstacle_length,
                     float min_obstacle_interval, float max_obstacle_interval,
                     uint obstacle_num)
      : height_dist_(min_obstacle_height, max_obstacle_height),
        length_dist_(min_obstacle_length, max_obstacle_length),
        interval_dist_(min_obstacle_interval, max_obstacle_interval),
        obstacle_num_(obstacle_num), corridor_length_(corridor_length),
        corridor_width_(corridor_width) {}

  void generateObstacles(raisim::World *world, std::mt19937 &gen) {
    corridor_ = world->addBox(corridor_length_, corridor_width_, 0.05, 0);
    corridor_->setPosition(corridor_length_ / 2 - 2, 0, -0.025);
    corridor_->setBodyType(raisim::BodyType::STATIC);
    float start_position = 1.0f;
    for (int i = 0; i < obstacle_num_; i++) {
      float interval = interval_dist_(gen);
      float height = height_dist_(gen);
      float length = length_dist_(gen);
      raisim::Box *obstacle = world->addBox(length, corridor_width_, height, 0);
      obstacle->setPosition(start_position + length / 2, 0, height / 2);
      obstacle->setBodyType(raisim::BodyType::STATIC);
      obstacles_.push_back(obstacle);
      start_position += interval;
    }
  }

  void regenerateObstacles(raisim::World *world, std::mt19937 &gen) {
    float start_position = 1.0f;
    for (int i = 0; i < obstacle_num_; i++) {
      world->removeObject(obstacles_[i]);
      float interval = interval_dist_(gen);
      float height = height_dist_(gen);
      float length = length_dist_(gen);
      obstacles_[i] = world->addBox(length, corridor_width_, height, 0);
      obstacles_[i]->setPosition(start_position + length / 2, 0, height / 2);
      obstacles_[i]->setBodyType(raisim::BodyType::STATIC);
      start_position += interval;
    }
  }

  void getObstacleHeight(float &min_height, float &max_height) {
    min_height = height_dist_.min();
    max_height = height_dist_.max();
  }

  void setObstacleHeight(float min_height, float max_height) {
    height_dist_ =
        std::uniform_real_distribution<float>(min_height, max_height);
  }

private:
  std::uniform_real_distribution<float> height_dist_;
  std::uniform_real_distribution<float> length_dist_;
  std::uniform_real_distribution<float> interval_dist_;
  std::vector<raisim::Box *> obstacles_;
  raisim::Box *corridor_;
  uint obstacle_num_;
  float corridor_width_, corridor_length_;
};
