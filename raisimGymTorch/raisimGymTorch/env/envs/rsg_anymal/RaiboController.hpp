//
// Created by donghoon on 8/11/22.
//

#ifndef _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#include "Common.hpp"
#include "Yaml.hpp"
#include "raisim/World.hpp"

namespace raisim {

class RaiboController {
public:
  inline bool create(raisim::World *world) {
    raibo_ = reinterpret_cast<raisim::ArticulatedSystem *>(
        world->getObject("robot"));
    gc_.setZero(raibo_->getGeneralizedCoordinateDim());
    gv_.setZero(raibo_->getDOF());
    jointVelocity_.resize(nJoints_);
    jointTorque_.setZero(nJoints_);

    /// Observation
    nominalJointConfig_.setZero(nJoints_);
    nominalJointConfig_ << 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0,
        0.580099, -1.195, 0, 0.580099, -1.195;
    jointTarget_.setZero(nJoints_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);

    actionMean_ << nominalJointConfig_;                     /// joint target
    actionStd_ << Eigen::VectorXd::Constant(nJoints_, 0.3); /// joint target

    obDouble_.setZero(obDim_);

    /// pd controller
    jointPgain_.setZero(gvDim_);
    jointPgain_.tail(nJoints_).setConstant(100.0);
    jointDgain_.setZero(gvDim_);
    jointDgain_.tail(nJoints_).setConstant(5.0);
    raibo_->setPGains(jointPgain_);
    raibo_->setDGains(jointDgain_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);

    stepData_.setZero(4);

    /// indices of links that should not make contact with ground
    footIndices_.push_back(raibo_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RH_SHANK"));
    RSFATAL_IF(std::any_of(footIndices_.begin(), footIndices_.end(),
                           [](int i) { return i < 0; }),
               "footIndices_ not found")

    /// indicies of the foot frame
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("LF_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("RF_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("LH_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("RH_S2F"));
    RSFATAL_IF(std::any_of(footFrameIndicies_.begin(), footFrameIndicies_.end(),
                           [](int i) { return i < 0; }),
               "footFrameIndicies_ not found")

    return true;
  };

  void reset(std::mt19937 &gen, std::uniform_real_distribution<double> &uniDist,
             raisim::HeightMap *heightMap) {
    double initial_heading = uniDist(gen) * 2 * M_PI;
    raisim::Mat<3, 3> rot;
    raisim::angleAxisToRotMat({0, 0, 1}, initial_heading, rot);
    raisim::Vec<4> quat;
    raisim::rotMatToQuat(rot, quat);

    gc_ << 0, 0, 0, quat.e(), nominalJointConfig_;
    raibo_->setState(gc_, gv_);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -std::numeric_limits<double>::infinity();
    for (auto &foot : footFrameIndicies_) {
      raibo_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition =
          heightMap->getHeight(footPosition[0], footPosition[1]) -
          footPosition[2];
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition
                              ? maxNecessaryShift
                              : terrainHeightMinusFootPosition;
    }

    double eps = 0.05;
    gc_ << 0, 0, maxNecessaryShift + eps, quat.e(), nominalJointConfig_;
    gv_ = gv_.setZero();
    raibo_->setState(gc_, gv_);
  }

  void updateStateVariables() {
    raibo_->getState(gc_, gv_);
    jointVelocity_ = gv_.tail(nJoints_);

    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    bodyLinVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
    bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);
    jointTorque_ = raibo_->getGeneralizedForce().e().tail(nJoints_);
  }

  bool advance(const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    jointTarget_ = action.cast<double>();

    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;

    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    return true;
  }

  void updateObservation(bool observationRandomization, std::mt19937 &gen,
                         std::normal_distribution<double> &normDist) {
    /// body orientation
    obDouble_.head(3) = baseRot_.e().row(2).transpose();
    /// body ang vel
    obDouble_.segment(3, 3) = bodyAngVel_;
    /// body lin vel
    obDouble_.segment(6, 3) = bodyLinVel_;
    /// joint pos
    obDouble_.segment(9, nJoints_) = gc_.tail(nJoints_);
    /// joint vel
    obDouble_.segment(21, nJoints_) = gv_.tail(nJoints_);
    /// command
    obDouble_.tail(3) = command_;

    if (observationRandomization) {
      for (int i = 0; i < 3; i++) {
        obDouble_(i) += 0.05 * normDist(gen);     /// orientation
        obDouble_(3 + i) += 0.22 * normDist(gen); /// body ang vel
      }
      obDouble_.head(3) /= obDouble_.head(3).norm();
      for (int i = 0; i < nJoints_; i++) {
        obDouble_(6 + i) += 0.055 * normDist(gen); /// joint pos
        obDouble_(18 + i) += 0.55 * normDist(gen); /// joint vel
      }
    }
  }

  void getObservation(Eigen::VectorXd &observation) { observation = obDouble_; }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(double, commandTrackingRewardCoeff,
              cfg["reward"]["command_tracking_reward_coeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, bodyContactRewardCoeff_,
              cfg["reward"]["body_contact_reward_coeff"])
  }

  inline void accumulateRewards(const double &cf, const double &terrainLevel) {
    double linearCommandTrackingReward = 0., angularCommandTrackingReward = 0.;
    linearCommandTrackingReward -=
        (command_.head(2) - bodyLinVel_.head(2)).squaredNorm();
    angularCommandTrackingReward -= pow((command_(2) - bodyAngVel_(2)), 2);
    commandTrackingReward_ +=
        (linearCommandTrackingReward + angularCommandTrackingReward) *
        commandTrackingRewardCoeff;

    torqueReward_ += torqueRewardCoeff_ * jointTorque_.squaredNorm();

    bodyContactReward_ = 0.;
    for (auto &contact : raibo_->getContacts()) {
      if (std::find(footIndices_.begin(), footIndices_.end(),
                    contact.getlocalBodyIndex()) == footIndices_.end()) {
        bodyContactReward_ = bodyContactRewardCoeff_;
        break;
      }
    }
  }

  [[nodiscard]] float getRewardSum(int steps) {
    int positiveRewardNum = 1;
    double positiveReward, negativeReward;
    stepData_[0] = commandTrackingReward_;
    stepData_[1] = torqueReward_;
    stepData_[2] = bodyContactReward_;
    stepData_ /= steps;
    positiveReward = stepData_.head(positiveRewardNum).sum();
    negativeReward = stepData_
                         .segment(positiveRewardNum,
                                  stepData_.size() - positiveRewardNum - 2)
                         .sum();
    stepData_[3] = positiveReward;
    stepData_[4] = negativeReward;

    commandTrackingReward_ = 0.;
    torqueReward_ = 0.;
    bodyContactReward_ = 0.;

    return float(positiveReward * std::exp(0.1 * negativeReward));
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    for (auto &contact : raibo_->getContacts())
      if (std::find(footIndices_.begin(), footIndices_.end(),
                    contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  inline void setCommand(const Eigen::Vector3d &command) { command_ = command; }

  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    gc = gc_.cast<float>();
    gv = gv_.cast<float>();
  }

  [[nodiscard]] inline const std::vector<std::string> &getStepDataTag() const {
    return stepDataTag_;
  }

  [[nodiscard]] inline const Eigen::VectorXd &getStepData() const {
    return stepData_;
  }

  std::map<std::string, float> getRewards() {
    std::map<std::string, float> rewards;
    rewards["command_tracking_reward"] = stepData_[0];
    rewards["torque_reward"] = stepData_[1];
    rewards["body_contact_reward"] = stepData_[2];
    return rewards;
  }

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  std::vector<size_t> footIndices_, footFrameIndicies_;
  Eigen::VectorXd nominalJointConfig_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t obDim_ = 36;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;

  // robot state variables
  Eigen::VectorXd gc_, gv_;
  Eigen::Vector3d bodyLinVel_,
      bodyAngVel_; /// body velocities are expressed in the body frame
  Eigen::VectorXd jointVelocity_;
  raisim::Mat<3, 3> baseRot_, controlRot_;

  // robot observation variables
  Eigen::VectorXd obDouble_;
  Eigen::Vector3d command_;

  // control variables
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_;
  Eigen::VectorXd jointTorque_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_;
  Eigen::VectorXd jointPgain_, jointDgain_;

  // reward variables
  double commandTrackingRewardCoeff = 0., commandTrackingReward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double bodyContactRewardCoeff_ = 0., bodyContactReward_ = 0.;
  double terminalRewardCoeff_ = 0.0;

  // exported data
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};

} // namespace raisim

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
