//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "../../RaisimGymEnv.hpp"
#include "RaiboController.hpp"
#include "RandomHeightMapGenerator.hpp"
#include <random>
#include <set>
#include <stdlib.h>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

public:
  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg,
                       bool visualizable)
      : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    std::string urdfPath;
    READ_YAML(std::string, urdfPath, cfg["urdf_path"]);
    raibo_ = world_->addArticulatedSystem(resourceDir_ + urdfPath);
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// create controller
    controller_.create(world_.get());
    controller_.setRewardConfig(cfg);

    /// add HeightMapGenerator
    // groundType_ = RandomHeightMapGenerator::GroundType(uniIntDist_(gen_));
    groundType_ = RandomHeightMapGenerator::GroundType::FLAT;
    groundTypeVector_ << 1.0, 0.0, 0.0;
    heightMap_ = terrainGenerator_.generateTerrain(world_.get(), groundType_,
                                                   gen_, uniDist_);
    footMaterialName_ = {"LF_FOOT_MATERIAL", "RF_FOOT_MATERIAL",
                         "LH_FOOT_MATERIAL", "RH_FOOT_MATERIAL"};

    READ_YAML(double, maxTime_, cfg["max_time"])

    /// Control parameters
    READ_YAML(double, maxForwardVel_, cfg["max_forward_velocity"])
    READ_YAML(double, maxCommandYaw_, cfg["max_command_yaw"])
    READ_YAML(double, maxAngularVel_, cfg["max_angular_velocity"])

    /// Curriculumm factors
    READ_YAML(int, iterPerCurriculumUpdate_,
              cfg["curriculum"]["iteration_per_update"])
    READ_YAML(int, iterPerTerrainCurriculumUpdate_,
              cfg["curriculum"]["iteration_per_terrain_update"])
    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])
    READ_YAML(double, terrainCurriculumFactor_,
              cfg["curriculum"]["terrain_initial_factor"])
    READ_YAML(double, terrainCurriculumDecayFactor_,
              cfg["curriculum"]["terrain_decay_factor"])

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = controller_.getObDim();
    actionDim_ = controller_.getActionDim();
    obDouble_.setZero(obDim_);

    /// create sensors
    depthSensor_ = raibo_->getSensorSet("d430_front")
                       ->getSensor<raisim::DepthCamera>("depth");
    depthSensor_->setMeasurementSource(
        raisim::Sensor::MeasurementSource::RAISIM);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      int port;
      READ_YAML(int, port, cfg["server_port"]);
      server_->launchServer(port);
      server_->focusOn(raibo_);
    }

    /// Just in case
    reset();
  }

  void init() final {}

  void reset() final {
    terrainRandomization();
    controller_.reset(gen_, uniDist_, heightMap_);
    controller_.updateStateVariables();
    resampleCommand();
  }

  void resampleCommand() {
    double vel_magnitude = maxForwardVel_ * uniDist_(gen_);
    double vel_direction = maxCommandYaw_ * 2 * (uniDist_(gen_) - 0.5);
    double ang_vel = maxAngularVel_ * 2 * (uniDist_(gen_) - 0.5);
    setCommand(vel_magnitude, vel_direction, ang_vel);
  }

  void setCommand(double vel_magnitude, double vel_direction, double ang_vel) {
    Eigen::Vector3d command;
    command << vel_magnitude * std::cos(vel_direction),
        vel_magnitude * std::sin(vel_direction), ang_vel;
    controller_.setCommand(command);
  }

  std::map<std::string, float> getRewards() final {
    return controller_.getRewards();
  }

  std::map<std::string, float> getTrainingInfo() {
    std::map<std::string, float> infos;
    infos["curriculum_factor"] = curriculumFactor_;
    infos["terrain_curriculum_factor"] = terrainCurriculumFactor_;
    return infos;
  }

  float step(const Eigen::Ref<EigenVec> &action) final {
    controller_.advance(action);

    int elapsed_sim_steps = 0;

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_)
        server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_)
        server_->unlockVisualizationServerMutex();
      controller_.updateStateVariables();
      controller_.accumulateRewards(curriculumFactor_, terrainLevel_);
      elapsed_sim_steps++;
    }
    if (uniDist_(gen_) <
        1 * control_dt_ /
            (maxTime_ + 1e-8)) { /// 1 times command change in 1 episode
      resampleCommand();
    }

    return controller_.getRewardSum(elapsed_sim_steps);
  }

  bool isTerminalState(float &terminalReward) {
    if (controller_.gc_.head(2).norm() > terrainGenerator_.size / 2.0) {
      return true;
    }
    return controller_.isTerminalState(terminalReward);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    controller_.updateObservation(false, gen_, normDist_);
    controller_.getObservation(obDouble_);
    ob = obDouble_.cast<float>();
  }

  void depthImage(Eigen::Ref<EigenRowMajorMat> image) final {
    depthSensor_->update(*world_);
    std::vector<float> im = depthSensor_->getDepthArray();
    image = Eigen::Map<EigenRowMajorMat>(im.data(), 64, 64);
  }

  void terrainRandomization() {
    for (auto &obj : world_->getObjList()) {
      if (obj->getName() != "robot") {
        world_->removeObject(obj);
      }
    }

    groundType_ = RandomHeightMapGenerator::GroundType(uniIntDist_(gen_));

    terrainLevel_ = terrainCurriculumFactor_ * uniDist_(gen_);
    Eigen::Vector4d terrainParams;
    terrainParams(0) = 2 * maxTime_ * maxForwardVel_ -
                       (2 * maxTime_ * maxForwardVel_ - 20) * terrainLevel_;
    if (groundType_ == RandomHeightMapGenerator::GroundType::HEIGHT_MAP) {
      terrainParams(1) = 0.2 + 0.8 * uniDist_(gen_); /// frequency ~ U(0.2, 1.0)
      terrainParams(2) = 0.2 + 1.2 * terrainLevel_;  /// amplitude 0.2 -> 1.4
      groundTypeVector_ << 1.0, 0.0, 0.0;
    } else if (groundType_ ==
               RandomHeightMapGenerator::GroundType::HEIGHT_MAP_DISCRETE) {
      terrainParams(1) = 0.2 + 1.0 * terrainLevel_; /// amplitude 0.2 -> 1.2
      terrainParams(2) =
          0.02 + 0.13 * uniDist_(gen_); /// step size ~ U(0.02, 0.15)
      groundTypeVector_ << 1.0, 0.0, 0.0;
    } else if (groundType_ == RandomHeightMapGenerator::GroundType::STEPS) {
      terrainParams(1) = 0.1 + 0.4 * uniDist_(gen_);  /// width ~ U(0.1, 0.5)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_; /// height 0.02 -> 0.18
      groundTypeVector_ << 0.0, 1.0, 0.0;
    } else if (groundType_ ==
               RandomHeightMapGenerator::GroundType::STEPS_INCLINE) {
      terrainParams(1) =
          0.02 + 0.16 * terrainLevel_; /// roughness ~ 0.02 -> 0.18
      terrainParams(2) = 0.02 + 0.07 * terrainLevel_; /// height 0.02 -> 0.09
      groundTypeVector_ << 1.0, 0.0, 0.0;
    } else if (groundType_ == RandomHeightMapGenerator::GroundType::STAIRS) {
      terrainParams(1) = 0.28 + 0.04 * uniDist_(gen_); /// width ~ U(0.28, 0.32)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_;  /// height 0.02 -> 0.18
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// slope 35 deg
      groundTypeVector_ << 0.0, 1.0, 0.0;
    } else if (groundType_ ==
               RandomHeightMapGenerator::GroundType::NOSING_STAIRS) {
      terrainParams(1) = 0.28 + 0.04 * uniDist_(gen_); /// width ~ U(0.28, 0.32)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_;  /// height 0.02 -> 0.18
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// slope 35 deg
      groundTypeVector_ << 0.0, 0.0, 1.0;
    }
    if (groundType_ == RandomHeightMapGenerator::GroundType::SLOPE) {
      terrainParams(1) = 0.2 + 0.6 * uniDist_(gen_); /// frequency ~ U(0.2, 0.8)
      terrainParams(2) = 0.2 + 0.4 * terrainLevel_;  /// amplitude 0.2 -> 0.6
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// 35deg
      groundTypeVector_ << 1.0, 0.0, 0.0;
    } else if (groundType_ == RandomHeightMapGenerator::GroundType::STAIRS3) {
      terrainParams(1) = 0.25 + 0.1 * uniDist_(gen_); /// width ~ U(0.25, 0.35)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_; /// height 0.02 -> 0.18
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// 35deg
      groundTypeVector_ << 0.0, 0.0, 1.0;
    } else if (groundType_ == RandomHeightMapGenerator::GroundType::TUK) {
      terrainParams(1) = 0.15 * terrainLevel_; /// height 0 -> 0.15
      groundTypeVector_ << 0.0, 0.0, 1.0;
    }
    terrainGenerator_.setTerrainParams(groundType_, terrainParams);
    heightMap_ = terrainGenerator_.generateTerrain(world_.get(), groundType_,
                                                   gen_, uniDist_);

    if (groundType_ == RandomHeightMapGenerator::GroundType::SLOPE) {
      double minFrictionCoeff = std::max(0.4, std::tan(terrainParams(3)) * 1.3);
      double interval = 2.0 - minFrictionCoeff;
      groundFrictionCoeff_ =
          minFrictionCoeff +
          interval * uniDist_(gen_); /// c_f ~ U(minFrictionCoeff, 2.0)
      world_->setDefaultMaterial(groundFrictionCoeff_, 0.0, 0.01);
      for (int i = 0; i < 4; i++) {
        world_->setMaterialPairProp(footMaterialName_[i],
                                    heightMap_->getCollisionObject()->material,
                                    groundFrictionCoeff_, 0.0, 0.01);
      }
    } else {
      groundFrictionCoeff_ =
          0.8 + terrainCurriculumFactor_ * 0.8 *
                    (uniDist_(gen_) - 0.5); /// c_f ~ U(0.4, 1.2)
      world_->setDefaultMaterial(groundFrictionCoeff_, 0.0, 0.01);
    }
  }

  void curriculumUpdate() {
    if (iter_ % iterPerCurriculumUpdate_ == 0) {
      curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    }
    if (iter_ % iterPerTerrainCurriculumUpdate_ == 0) {
      terrainCurriculumFactor_ =
          std::pow(terrainCurriculumFactor_, terrainCurriculumDecayFactor_);
    }
    iter_++;
  }

  void setSeed(int seed) {
    gen_.seed(seed);
    terrainGenerator_.setSeed(seed);
  }

private:
  bool visualizable_ = false;
  raisim::ArticulatedSystem *raibo_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  raisim::DepthCamera *depthSensor_;
  double terrainCurriculumFactor_, terrainCurriculumDecayFactor_,
      terrainLevel_ = 0.;
  double curriculumFactor_, curriculumDecayFactor_, maxForwardVel_,
      maxCommandYaw_, maxAngularVel_;
  double maxTime_;
  Eigen::Vector3d groundTypeVector_;
  RandomHeightMapGenerator terrainGenerator_;
  RandomHeightMapGenerator::GroundType groundType_;
  raisim::HeightMap *heightMap_;
  RaiboController controller_;
  double groundFrictionCoeff_;
  std::vector<std::string> footMaterialName_;
  int iter_ = 0;
  int iterPerCurriculumUpdate_ = 1;
  int iterPerTerrainCurriculumUpdate_ = 1;
  static constexpr int terrainTypeNum_ = 9;

  /// these variables are not in use. They are placed to show you how to
  /// create a random number sampler.
  thread_local static std::mt19937 gen_;

  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::uniform_int_distribution<int> uniIntDist_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double>
    raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double>
    raisim::ENVIRONMENT::uniDist_(0., 1.);
thread_local std::uniform_int_distribution<int>
    raisim::ENVIRONMENT::uniIntDist_(0, raisim::ENVIRONMENT::terrainTypeNum_);
} // namespace raisim
