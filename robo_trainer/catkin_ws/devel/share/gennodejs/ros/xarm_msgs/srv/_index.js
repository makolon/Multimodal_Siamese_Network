
"use strict";

let GripperMove = require('./GripperMove.js')
let GripperConfig = require('./GripperConfig.js')
let SetFloat32 = require('./SetFloat32.js')
let GetAnalogIO = require('./GetAnalogIO.js')
let SetString = require('./SetString.js')
let GripperState = require('./GripperState.js')
let SetDigitalIO = require('./SetDigitalIO.js')
let GetControllerDigitalIO = require('./GetControllerDigitalIO.js')
let SetMultipleInts = require('./SetMultipleInts.js')
let MoveAxisAngle = require('./MoveAxisAngle.js')
let SetInt16 = require('./SetInt16.js')
let SetToolModbus = require('./SetToolModbus.js')
let PlayTraj = require('./PlayTraj.js')
let SetControllerAnalogIO = require('./SetControllerAnalogIO.js')
let GetDigitalIO = require('./GetDigitalIO.js')
let MoveVelo = require('./MoveVelo.js')
let ClearErr = require('./ClearErr.js')
let GetErr = require('./GetErr.js')
let Move = require('./Move.js')
let SetAxis = require('./SetAxis.js')
let ConfigToolModbus = require('./ConfigToolModbus.js')
let TCPOffset = require('./TCPOffset.js')
let SetLoad = require('./SetLoad.js')

module.exports = {
  GripperMove: GripperMove,
  GripperConfig: GripperConfig,
  SetFloat32: SetFloat32,
  GetAnalogIO: GetAnalogIO,
  SetString: SetString,
  GripperState: GripperState,
  SetDigitalIO: SetDigitalIO,
  GetControllerDigitalIO: GetControllerDigitalIO,
  SetMultipleInts: SetMultipleInts,
  MoveAxisAngle: MoveAxisAngle,
  SetInt16: SetInt16,
  SetToolModbus: SetToolModbus,
  PlayTraj: PlayTraj,
  SetControllerAnalogIO: SetControllerAnalogIO,
  GetDigitalIO: GetDigitalIO,
  MoveVelo: MoveVelo,
  ClearErr: ClearErr,
  GetErr: GetErr,
  Move: Move,
  SetAxis: SetAxis,
  ConfigToolModbus: ConfigToolModbus,
  TCPOffset: TCPOffset,
  SetLoad: SetLoad,
};
