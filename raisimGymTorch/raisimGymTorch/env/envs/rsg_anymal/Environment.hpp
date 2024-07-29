//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "../../RaisimGymEnv.hpp"
#include "DreamTeam.hpp"
#include <set>
#include <stdlib.h>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

public:
  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg,
                       bool visualizable)
      : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable),
        normDist_(0, 1),
        obstacleCourse_(100.0, 4.0, 0.0, 0.1, 0.3, 2.0, 0.0, 2.0, 30) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    unsigned int seed;
    READ_YAML(int, seed, cfg["seed"]);
    gen_ = std::mt19937(seed);
    std::cout << "Environment seed: " << seed << std::endl;

    /// add objects
    std::string urdfPath;
    READ_YAML(std::string, urdfPath, cfg["urdf_path"]);
    anymal_ = world_->addArticulatedSystem(resourceDir_ + urdfPath);
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    // world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4,
        -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    double p_gain;
    double d_gain;
    READ_YAML(double, p_gain, cfg_["p_gain"])
    READ_YAML(double, d_gain, cfg_["d_gain"])
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(p_gain);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(d_gain);

    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// create sensors
    depthSensor_ = anymal_->getSensorSet("depth_camera")
                       ->getSensor<raisim::DepthCamera>("depth");
    depthSensor_->setMeasurementSource(
        raisim::Sensor::MeasurementSource::RAISIM);

    /// generate obstacles
    obstacleCourse_.generateObstacles(world_.get(), gen_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std,
              cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile(cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      int port;
      READ_YAML(int, port, cfg["server_port"]);
      server_->launchServer(port);
      server_->focusOn(anymal_);
    }
  }

  void init() final {}

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    obstacleCourse_.regenerateObstacles(world_.get(), gen_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec> &action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_)
        server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_)
        server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    // rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));
    float reward = rewards_.sum();
    reward += 0.3 * std::min(2.0, gv_[0]);
    // reward is zero when falling
    if (gc_[2] < 0.01) {
      reward = -1;
    }
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    if (rot.e().col(2).dot(Eigen::Vector3d(0, 0, 1)) < 0.5) {
      reward = -1;
    }

    return reward;
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2],                 /// body height
        rot.e().row(2).transpose(),      /// body orientation
        gc_.tail(12),                    /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12);                    /// joint velocity
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float &terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for (auto &contact : anymal_->getContacts())
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void depthImage(Eigen::Ref<EigenRowMajorMat> image) final {
    depthSensor_->update(*world_);
    std::vector<float> im = depthSensor_->getDepthArray();
    image = Eigen::Map<Eigen::MatrixXf>(im.data(), 64, 64);
  }

  void curriculumUpdate() {
    float min_height = 0.0;
    float max_height = 0.0;
    obstacleCourse_.getObstacleHeight(min_height, max_height);
    if (max_height < 0.4) {
      obstacleCourse_.setObstacleHeight(min_height, max_height + 0.0001);
    }
  }

private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem *anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  raisim::DepthCamera *depthSensor_;
  DreamTeam obstacleCourse_;

  /// these variables are not in use. They are placed to show you how to create
  /// a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

} // namespace raisim
