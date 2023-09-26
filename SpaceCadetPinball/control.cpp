#include "pch.h"
#include "control.h"

#include "midi.h"
#include "pb.h"
#include "TBall.h"
#include "TBlocker.h"
#include "TBumper.h"
#include "TComponentGroup.h"
#include "TFlagSpinner.h"
#include "TLight.h"
#include "TLightBargraph.h"
#include "TLightGroup.h"
#include "TLightRollover.h"
#include "TPinballTable.h"
#include "TPopupTarget.h"
#include "TSink.h"
#include "TSoloTarget.h"
#include "TSound.h"
#include "TTripwire.h"
#include "TDrain.h"
#include "TFlipper.h"
#include "TGate.h"
#include "THole.h"
#include "TKickback.h"
#include "TKickout.h"
#include "TOneway.h"
#include "TRamp.h"
#include "TPlunger.h"
#include "TWall.h"
#include "TTextBox.h"
#include "translations.h"

int control_bump_scores1[] = {500, 1000, 1500, 2000};
int control_roll_scores1[] = {2000};
int control_bump_scores2[] = {1500, 2500, 3500, 4500};
int control_roll_scores2[] = {500};
int control_rebo_score1[] = {500};
int control_oneway4_score1[6] = {15000, 30000, 75000, 30000, 15000, 7500};
int control_ramp_score1[1] = {5000};
int control_roll_score1[1] = {20000};
int control_roll_score2[2] = {5000, 25000};
int control_roll_score3[1] = {10000};
int control_roll_score4[1] = {500};
int control_flag_score1[2] = {500, 2500};
int control_kickout_score1[5] = {10000, 0, 20000, 50000, 150000};
int control_sink_score1[3] = {2500, 5000, 7500};
int control_target_score1[2] = {500, 5000};
int control_target_score2[3] = {1500, 10000, 50000};
int control_target_score3[2] = {500, 1500};
int control_target_score4[1] = {750};
int control_target_score5[1] = {1000};
int control_target_score6[1] = {750};
int control_target_score7[1] = {750};
int control_roll_score5[1] = {10000};
int control_kickout_score2[1] = {20000};
int control_kickout_score3[1] = {50000};


component_tag<TComponentGroup> control_attack_bump_tag = {"attack_bumpers"};
component_tag<TComponentGroup> control_launch_bump_tag = {"launch_bumpers"};
component_tag<TBlocker> control_block1_tag = {"v_bloc1"};
component_tag<TBumper> control_bump1_tag = {"a_bump1"};
component_tag<TBumper> control_bump2_tag = {"a_bump2"};
component_tag<TBumper> control_bump3_tag = {"a_bump3"};
component_tag<TBumper> control_bump4_tag = {"a_bump4"};
component_tag<TBumper> control_bump5_tag = {"a_bump5"};
component_tag<TBumper> control_bump6_tag = {"a_bump6"};
component_tag<TBumper> control_bump7_tag = {"a_bump7"};
component_tag<TDrain> control_drain_tag = {"drain"};
component_tag<TFlagSpinner> control_flag1_tag = {"a_flag1"};
component_tag<TFlagSpinner> control_flag2_tag = {"a_flag2"};
component_tag<TFlipper> control_flip1_tag = {"a_flip1"};
component_tag<TFlipper> control_flip2_tag = {"a_flip2"};
component_tag<TLightBargraph> control_fuel_bargraph_tag = {"fuel_bargraph"};
component_tag<TGate> control_gate1_tag = {"v_gate1"};
component_tag<TGate> control_gate2_tag = {"v_gate2"};
component_tag<TTextBox> control_info_text_box_tag = {"info_text_box"};
component_tag<TKickback> control_kicker1_tag = {"a_kick1"};
component_tag<TKickback> control_kicker2_tag = {"a_kick2"};
component_tag<TKickout> control_kickout1_tag = {"a_kout1"};
component_tag<TKickout> control_kickout2_tag = {"a_kout2"};
component_tag<TKickout> control_kickout3_tag = {"a_kout3"};
component_tag<TLight> control_lite1_tag = {"lite1"};
component_tag<TLight> control_lite2_tag = {"lite2"};
component_tag<TLight> control_lite3_tag = {"lite3"};
component_tag<TLight> control_lite4_tag = {"lite4"};
component_tag<TLight> control_lite5_tag = {"lite5"};
component_tag<TLight> control_lite6_tag = {"lite6"};
component_tag<TLight> control_lite7_tag = {"lite7"};
component_tag<TLight> control_lite8_tag = {"lite8"};
component_tag<TLight> control_lite9_tag = {"lite9"};
component_tag<TLight> control_lite10_tag = {"lite10"};
component_tag<TLight> control_lite11_tag = {"lite11"};
component_tag<TLight> control_lite12_tag = {"lite12"};
component_tag<TLight> control_lite13_tag = {"lite13"};
component_tag<TLight> control_lite16_tag = {"lite16"};
component_tag<TLight> control_lite17_tag = {"lite17"};
component_tag<TLight> control_lite18_tag = {"lite18"};
component_tag<TLight> control_lite19_tag = {"lite19"};
component_tag<TLight> control_lite20_tag = {"lite20"};
component_tag<TLight> control_lite21_tag = {"lite21"};
component_tag<TLight> control_lite22_tag = {"lite22"};
component_tag<TLight> control_lite23_tag = {"lite23"};
component_tag<TLight> control_lite24_tag = {"lite24"};
component_tag<TLight> control_lite25_tag = {"lite25"};
component_tag<TLight> control_lite26_tag = {"lite26"};
component_tag<TLight> control_lite27_tag = {"lite27"};
component_tag<TLight> control_lite28_tag = {"lite28"};
component_tag<TLight> control_lite29_tag = {"lite29"};
component_tag<TLight> control_lite30_tag = {"lite30"};
component_tag<TLight> control_lite38_tag = {"lite38"};
component_tag<TLight> control_lite39_tag = {"lite39"};
component_tag<TLight> control_lite40_tag = {"lite40"};
component_tag<TLight> control_lite54_tag = {"lite54"};
component_tag<TLight> control_lite55_tag = {"lite55"};
component_tag<TLight> control_lite56_tag = {"lite56"};
component_tag<TLight> control_lite58_tag = {"lite58"};
component_tag<TLight> control_lite59_tag = {"lite59"};
component_tag<TLight> control_lite60_tag = {"lite60"};
component_tag<TLight> control_lite61_tag = {"lite61"};
component_tag<TLight> control_lite62_tag = {"lite62"};
component_tag<TLight> control_lite67_tag = {"lite67"};
component_tag<TLight> control_lite68_tag = {"lite68"};
component_tag<TLight> control_lite69_tag = {"lite69"};
component_tag<TLight> control_lite70_tag = {"lite70"};
component_tag<TLight> control_lite71_tag = {"lite71"};
component_tag<TLight> control_lite72_tag = {"lite72"};
component_tag<TLight> control_lite77_tag = {"lite77"};
component_tag<TLight> control_lite84_tag = {"lite84"};
component_tag<TLight> control_lite85_tag = {"lite85"};
component_tag<TLight> control_lite101_tag = {"lite101"};
component_tag<TLight> control_lite102_tag = {"lite102"};
component_tag<TLight> control_lite103_tag = {"lite103"};
component_tag<TLight> control_lite104_tag = {"lite104"};
component_tag<TLight> control_lite105_tag = {"lite105"};
component_tag<TLight> control_lite106_tag = {"lite106"};
component_tag<TLight> control_lite107_tag = {"lite107"};
component_tag<TLight> control_lite108_tag = {"lite108"};
component_tag<TLight> control_lite109_tag = {"lite109"};
component_tag<TLight> control_lite110_tag = {"lite110"};
component_tag<TLight> control_lite130_tag = {"lite130"};
component_tag<TLight> control_lite131_tag = {"lite131"};
component_tag<TLight> control_lite132_tag = {"lite132"};
component_tag<TLight> control_lite133_tag = {"lite133"};
component_tag<TLight> control_lite169_tag = {"lite169"};
component_tag<TLight> control_lite170_tag = {"lite170"};
component_tag<TLight> control_lite171_tag = {"lite171"};
component_tag<TLight> control_lite195_tag = {"lite195"};
component_tag<TLight> control_lite196_tag = {"lite196"};
component_tag<TLight> control_lite198_tag = {"lite198"};
component_tag<TLight> control_lite199_tag = {"lite199"};
component_tag<TLight> control_lite200_tag = {"lite200"};
component_tag<TLight> control_lite300_tag = {"lite300"};
component_tag<TLight> control_lite301_tag = {"lite301"};
component_tag<TLight> control_lite302_tag = {"lite302"};
component_tag<TLight> control_lite303_tag = {"lite303"};
component_tag<TLight> control_lite304_tag = {"lite304"};
component_tag<TLight> control_lite305_tag = {"lite305"};
component_tag<TLight> control_lite306_tag = {"lite306"};
component_tag<TLight> control_lite307_tag = {"lite307"};
component_tag<TLight> control_lite308_tag = {"lite308"};
component_tag<TLight> control_lite309_tag = {"lite309"};
component_tag<TLight> control_lite310_tag = {"lite310"};
component_tag<TLight> control_lite311_tag = {"lite311"};
component_tag<TLight> control_lite312_tag = {"lite312"};
component_tag<TLight> control_lite313_tag = {"lite313"};
component_tag<TLight> control_lite314_tag = {"lite314"};
component_tag<TLight> control_lite315_tag = {"lite315"};
component_tag<TLight> control_lite316_tag = {"lite316"};
component_tag<TLight> control_lite317_tag = {"lite317"};
component_tag<TLight> control_lite318_tag = {"lite318"};
component_tag<TLight> control_lite319_tag = {"lite319"};
component_tag<TLight> control_lite320_tag = {"lite320"};
component_tag<TLight> control_lite321_tag = {"lite321"};
component_tag<TLight> control_lite322_tag = {"lite322"};
component_tag<TLight> control_literoll179_tag = {"literoll179"};
component_tag<TLight> control_literoll180_tag = {"literoll180"};
component_tag<TLight> control_literoll181_tag = {"literoll181"};
component_tag<TLight> control_literoll182_tag = {"literoll182"};
component_tag<TLight> control_literoll183_tag = {"literoll183"};
component_tag<TLight> control_literoll184_tag = {"literoll184"};
component_tag<TLightGroup> control_middle_circle_tag = {"middle_circle"};
component_tag<TLightGroup> control_lchute_tgt_lights_tag = {"lchute_tgt_lights"};
component_tag<TLightGroup> control_l_trek_lights_tag = {"l_trek_lights"};
component_tag<TLightGroup> control_goal_lights_tag = {"goal_lights"};
component_tag<TLightGroup> control_hyper_lights_tag = {"hyperspace_lights"};
component_tag<TLightGroup> control_bmpr_inc_lights_tag = {"bmpr_inc_lights"};
component_tag<TLightGroup> control_bpr_solotgt_lights_tag = {"bpr_solotgt_lights"};
component_tag<TLightGroup> control_bsink_arrow_lights_tag = {"bsink_arrow_lights"};
component_tag<TLightGroup> control_bumber_target_lights_tag = {"bumper_target_lights"};
component_tag<TLightGroup> control_outer_circle_tag = {"outer_circle"};
component_tag<TLightGroup> control_r_trek_lights_tag = {"r_trek_lights"};
component_tag<TLightGroup> control_ramp_bmpr_inc_lights_tag = {"ramp_bmpr_inc_lights"};
component_tag<TLightGroup> control_ramp_tgt_lights_tag = {"ramp_tgt_lights"};
component_tag<TLightGroup> control_skill_shot_lights_tag = {"skill_shot_lights"};
component_tag<TLightGroup> control_top_circle_tgt_lights_tag = {"top_circle_tgt_lights"};
component_tag<TLightGroup> control_top_target_lights_tag = {"top_target_lights"};
component_tag<TLightGroup> control_worm_hole_lights_tag = {"worm_hole_lights"};
component_tag<TTextBox> control_mission_text_box_tag = {"mission_text_box"};
component_tag<TOneway> control_oneway1_tag = {"s_onewy1"};
component_tag<TOneway> control_oneway4_tag = {"s_onewy4"};
component_tag<TOneway> control_oneway10_tag = {"s_onewy10"};
component_tag<TPlunger> control_plunger_tag = {"plunger"};
component_tag<THole> control_ramp_hole_tag = {"ramp_hole"};
component_tag<TRamp> control_ramp_tag = {"ramp"};
component_tag<TWall> control_rebo1_tag = {"v_rebo1"};
component_tag<TWall> control_rebo2_tag = {"v_rebo2"};
component_tag<TWall> control_rebo3_tag = {"v_rebo3"};
component_tag<TWall> control_rebo4_tag = {"v_rebo4"};
component_tag<TRollover> control_roll1_tag = {"a_roll1"};
component_tag<TRollover> control_roll2_tag = {"a_roll2"};
component_tag<TRollover> control_roll3_tag = {"a_roll3"};
component_tag<TRollover> control_roll4_tag = {"a_roll4"};
component_tag<TRollover> control_roll5_tag = {"a_roll5"};
component_tag<TRollover> control_roll6_tag = {"a_roll6"};
component_tag<TRollover> control_roll7_tag = {"a_roll7"};
component_tag<TRollover> control_roll8_tag = {"a_roll8"};
component_tag<TLightRollover> control_roll9_tag = {"a_roll9"};
component_tag<TRollover> control_roll110_tag = {"a_roll110"};
component_tag<TRollover> control_roll111_tag = {"a_roll111"};
component_tag<TRollover> control_roll112_tag = {"a_roll112"};
component_tag<TRollover> control_roll179_tag = {"a_roll179"};
component_tag<TRollover> control_roll180_tag = {"a_roll180"};
component_tag<TRollover> control_roll181_tag = {"a_roll181"};
component_tag<TRollover> control_roll182_tag = {"a_roll182"};
component_tag<TRollover> control_roll183_tag = {"a_roll183"};
component_tag<TRollover> control_roll184_tag = {"a_roll184"};
component_tag<TSink> control_sink1_tag = {"v_sink1"};
component_tag<TSink> control_sink2_tag = {"v_sink2"};
component_tag<TSink> control_sink3_tag = {"v_sink3"};
component_tag<TSink> control_sink7_tag = {"v_sink7"};
component_tag<TSound> control_soundwave3_tag = {"soundwave3"};
component_tag<TSound> control_soundwave7_tag = {"soundwave7"};
component_tag<TSound> control_soundwave8_tag = {"soundwave8"};
component_tag<TSound> control_soundwave9_tag = {"soundwave9"};
component_tag<TSound> control_soundwave10_tag = {"soundwave10"};
component_tag<TSound> control_soundwave14_1_tag = {"soundwave14"};
component_tag<TSound> control_soundwave14_2_tag = {"soundwave14"};
component_tag<TSound> control_soundwave21_tag = {"soundwave21"};
component_tag<TSound> control_soundwave23_tag = {"soundwave23"};
component_tag<TSound> control_soundwave24_tag = {"soundwave24"};
component_tag<TSound> control_soundwave25_tag = {"soundwave25"};
component_tag<TSound> control_soundwave26_tag = {"soundwave26"};
component_tag<TSound> control_soundwave27_tag = {"soundwave27"};
component_tag<TSound> control_soundwave28_tag = {"soundwave28"};
component_tag<TSound> control_soundwave30_tag = {"soundwave30"};
component_tag<TSound> control_soundwave35_1_tag = {"soundwave35"};
component_tag<TSound> control_soundwave35_2_tag = {"soundwave35"};
component_tag<TSound> control_soundwave36_1_tag = {"soundwave36"};
component_tag<TSound> control_soundwave36_2_tag = {"soundwave36"};
component_tag<TSound> control_soundwave38_tag = {"soundwave38"};
component_tag<TSound> control_soundwave39_tag = {"soundwave39"};
component_tag<TSound> control_soundwave40_tag = {"soundwave40"};
component_tag<TSound> control_soundwave41_tag = {"soundwave41"};
component_tag<TSound> control_soundwave44_tag = {"soundwave44"};
component_tag<TSound> control_soundwave45_tag = {"soundwave45"};
component_tag<TSound> control_soundwave46_tag = {"soundwave46"};
component_tag<TSound> control_soundwave47_tag = {"soundwave47"};
component_tag<TSound> control_soundwave48_tag = {"soundwave48"};
component_tag<TSound> control_soundwave49D_tag = {"soundwave49D"};
component_tag<TSound> control_soundwave50_1_tag = {"soundwave50"};
component_tag<TSound> control_soundwave50_2_tag = {"soundwave50"};
component_tag<TSound> control_soundwave52_tag = {"soundwave52"};
component_tag<TSound> control_soundwave59_tag = {"soundwave59"};
component_tag<TPopupTarget> control_target1_tag = {"a_targ1"};
component_tag<TPopupTarget> control_target2_tag = {"a_targ2"};
component_tag<TPopupTarget> control_target3_tag = {"a_targ3"};
component_tag<TPopupTarget> control_target4_tag = {"a_targ4"};
component_tag<TPopupTarget> control_target5_tag = {"a_targ5"};
component_tag<TPopupTarget> control_target6_tag = {"a_targ6"};
component_tag<TPopupTarget> control_target7_tag = {"a_targ7"};
component_tag<TPopupTarget> control_target8_tag = {"a_targ8"};
component_tag<TPopupTarget> control_target9_tag = {"a_targ9"};
component_tag<TSoloTarget> control_target10_tag = {"a_targ10"};
component_tag<TSoloTarget> control_target11_tag = {"a_targ11"};
component_tag<TSoloTarget> control_target12_tag = {"a_targ12"};
component_tag<TSoloTarget> control_target13_tag = {"a_targ13"};
component_tag<TSoloTarget> control_target14_tag = {"a_targ14"};
component_tag<TSoloTarget> control_target15_tag = {"a_targ15"};
component_tag<TSoloTarget> control_target16_tag = {"a_targ16"};
component_tag<TSoloTarget> control_target17_tag = {"a_targ17"};
component_tag<TSoloTarget> control_target18_tag = {"a_targ18"};
component_tag<TSoloTarget> control_target19_tag = {"a_targ19"};
component_tag<TSoloTarget> control_target20_tag = {"a_targ20"};
component_tag<TSoloTarget> control_target21_tag = {"a_targ21"};
component_tag<TSoloTarget> control_target22_tag = {"a_targ22"};
component_tag<TTripwire> control_trip1_tag = {"s_trip1"};
component_tag<TTripwire> control_trip2_tag = {"s_trip2"};
component_tag<TTripwire> control_trip3_tag = {"s_trip3"};
component_tag<TTripwire> control_trip4_tag = {"s_trip4"};
component_tag<TTripwire> control_trip5_tag = {"s_trip5"};


// Component shortcuts for easier access without indirection through tags 
TComponentGroup*& attack_bump = control_attack_bump_tag.Component;
TComponentGroup*& launch_bump = control_launch_bump_tag.Component;
TBlocker*& block1 = control_block1_tag.Component;
TBumper*& bump1 = control_bump1_tag.Component;
TBumper*& bump2 = control_bump2_tag.Component;
TBumper*& bump3 = control_bump3_tag.Component;
TBumper*& bump4 = control_bump4_tag.Component;
TBumper*& bump5 = control_bump5_tag.Component;
TBumper*& bump6 = control_bump6_tag.Component;
TBumper*& bump7 = control_bump7_tag.Component;
TDrain*& drain = control_drain_tag.Component;
TFlagSpinner*& flag1 = control_flag1_tag.Component;
TFlagSpinner*& flag2 = control_flag2_tag.Component;
TFlipper*& flip1 = control_flip1_tag.Component;
TFlipper*& flip2 = control_flip2_tag.Component;
TLightBargraph*& fuel_bargraph = control_fuel_bargraph_tag.Component;
TGate*& gate1 = control_gate1_tag.Component;
TGate*& gate2 = control_gate2_tag.Component;
TTextBox*& info_text_box = control_info_text_box_tag.Component;
TKickback*& kicker1 = control_kicker1_tag.Component;
TKickback*& kicker2 = control_kicker2_tag.Component;
TKickout*& kickout1 = control_kickout1_tag.Component;
TKickout*& kickout2 = control_kickout2_tag.Component;
TKickout*& kickout3 = control_kickout3_tag.Component;
TLight*& lite1 = control_lite1_tag.Component;
TLight*& lite2 = control_lite2_tag.Component;
TLight*& lite3 = control_lite3_tag.Component;
TLight*& lite4 = control_lite4_tag.Component;
TLight*& lite5 = control_lite5_tag.Component;
TLight*& lite6 = control_lite6_tag.Component;
TLight*& lite7 = control_lite7_tag.Component;
TLight*& lite8 = control_lite8_tag.Component;
TLight*& lite9 = control_lite9_tag.Component;
TLight*& lite10 = control_lite10_tag.Component;
TLight*& lite11 = control_lite11_tag.Component;
TLight*& lite12 = control_lite12_tag.Component;
TLight*& lite13 = control_lite13_tag.Component;
TLight*& lite16 = control_lite16_tag.Component;
TLight*& lite17 = control_lite17_tag.Component;
TLight*& lite18 = control_lite18_tag.Component;
TLight*& lite19 = control_lite19_tag.Component;
TLight*& lite20 = control_lite20_tag.Component;
TLight*& lite21 = control_lite21_tag.Component;
TLight*& lite22 = control_lite22_tag.Component;
TLight*& lite23 = control_lite23_tag.Component;
TLight*& lite24 = control_lite24_tag.Component;
TLight*& lite25 = control_lite25_tag.Component;
TLight*& lite26 = control_lite26_tag.Component;
TLight*& lite27 = control_lite27_tag.Component;
TLight*& lite28 = control_lite28_tag.Component;
TLight*& lite29 = control_lite29_tag.Component;
TLight*& lite30 = control_lite30_tag.Component;
TLight*& lite38 = control_lite38_tag.Component;
TLight*& lite39 = control_lite39_tag.Component;
TLight*& lite40 = control_lite40_tag.Component;
TLight*& lite54 = control_lite54_tag.Component;
TLight*& lite55 = control_lite55_tag.Component;
TLight*& lite56 = control_lite56_tag.Component;
TLight*& lite58 = control_lite58_tag.Component;
TLight*& lite59 = control_lite59_tag.Component;
TLight*& lite60 = control_lite60_tag.Component;
TLight*& lite61 = control_lite61_tag.Component;
TLight*& lite62 = control_lite62_tag.Component;
TLight*& lite67 = control_lite67_tag.Component;
TLight*& lite68 = control_lite68_tag.Component;
TLight*& lite69 = control_lite69_tag.Component;
TLight*& lite70 = control_lite70_tag.Component;
TLight*& lite71 = control_lite71_tag.Component;
TLight*& lite72 = control_lite72_tag.Component;
TLight*& lite77 = control_lite77_tag.Component;
TLight*& lite84 = control_lite84_tag.Component;
TLight*& lite85 = control_lite85_tag.Component;
TLight*& lite101 = control_lite101_tag.Component;
TLight*& lite102 = control_lite102_tag.Component;
TLight*& lite103 = control_lite103_tag.Component;
TLight*& lite104 = control_lite104_tag.Component;
TLight*& lite105 = control_lite105_tag.Component;
TLight*& lite106 = control_lite106_tag.Component;
TLight*& lite107 = control_lite107_tag.Component;
TLight*& lite108 = control_lite108_tag.Component;
TLight*& lite109 = control_lite109_tag.Component;
TLight*& lite110 = control_lite110_tag.Component;
TLight*& lite130 = control_lite130_tag.Component;
TLight*& lite131 = control_lite131_tag.Component;
TLight*& lite132 = control_lite132_tag.Component;
TLight*& lite133 = control_lite133_tag.Component;
TLight*& lite169 = control_lite169_tag.Component;
TLight*& lite170 = control_lite170_tag.Component;
TLight*& lite171 = control_lite171_tag.Component;
TLight*& lite195 = control_lite195_tag.Component;
TLight*& lite196 = control_lite196_tag.Component;
TLight*& lite198 = control_lite198_tag.Component;
TLight*& lite199 = control_lite199_tag.Component;
TLight*& lite200 = control_lite200_tag.Component;
TLight*& lite300 = control_lite300_tag.Component;
TLight*& lite301 = control_lite301_tag.Component;
TLight*& lite302 = control_lite302_tag.Component;
TLight*& lite303 = control_lite303_tag.Component;
TLight*& lite304 = control_lite304_tag.Component;
TLight*& lite305 = control_lite305_tag.Component;
TLight*& lite306 = control_lite306_tag.Component;
TLight*& lite307 = control_lite307_tag.Component;
TLight*& lite308 = control_lite308_tag.Component;
TLight*& lite309 = control_lite309_tag.Component;
TLight*& lite310 = control_lite310_tag.Component;
TLight*& lite311 = control_lite311_tag.Component;
TLight*& lite312 = control_lite312_tag.Component;
TLight*& lite313 = control_lite313_tag.Component;
TLight*& lite314 = control_lite314_tag.Component;
TLight*& lite315 = control_lite315_tag.Component;
TLight*& lite316 = control_lite316_tag.Component;
TLight*& lite317 = control_lite317_tag.Component;
TLight*& lite318 = control_lite318_tag.Component;
TLight*& lite319 = control_lite319_tag.Component;
TLight*& lite320 = control_lite320_tag.Component;
TLight*& lite321 = control_lite321_tag.Component;
TLight*& lite322 = control_lite322_tag.Component;
TLight*& literoll179 = control_literoll179_tag.Component;
TLight*& literoll180 = control_literoll180_tag.Component;
TLight*& literoll181 = control_literoll181_tag.Component;
TLight*& literoll182 = control_literoll182_tag.Component;
TLight*& literoll183 = control_literoll183_tag.Component;
TLight*& literoll184 = control_literoll184_tag.Component;
TLightGroup*& middle_circle = control_middle_circle_tag.Component;
TLightGroup*& lchute_tgt_lights = control_lchute_tgt_lights_tag.Component;
TLightGroup*& l_trek_lights = control_l_trek_lights_tag.Component;
TLightGroup*& goal_lights = control_goal_lights_tag.Component;
TLightGroup*& hyper_lights = control_hyper_lights_tag.Component;
TLightGroup*& bmpr_inc_lights = control_bmpr_inc_lights_tag.Component;
TLightGroup*& bpr_solotgt_lights = control_bpr_solotgt_lights_tag.Component;
TLightGroup*& bsink_arrow_lights = control_bsink_arrow_lights_tag.Component;
TLightGroup*& bumber_target_lights = control_bumber_target_lights_tag.Component;
TLightGroup*& outer_circle = control_outer_circle_tag.Component;
TLightGroup*& r_trek_lights = control_r_trek_lights_tag.Component;
TLightGroup*& ramp_bmpr_inc_lights = control_ramp_bmpr_inc_lights_tag.Component;
TLightGroup*& ramp_tgt_lights = control_ramp_tgt_lights_tag.Component;
TLightGroup*& skill_shot_lights = control_skill_shot_lights_tag.Component;
TLightGroup*& top_circle_tgt_lights = control_top_circle_tgt_lights_tag.Component;
TLightGroup*& top_target_lights = control_top_target_lights_tag.Component;
TLightGroup*& worm_hole_lights = control_worm_hole_lights_tag.Component;
TTextBox*& mission_text_box = control_mission_text_box_tag.Component;
TOneway*& oneway1 = control_oneway1_tag.Component;
TOneway*& oneway4 = control_oneway4_tag.Component;
TOneway*& oneway10 = control_oneway10_tag.Component;
TPlunger*& plunger = control_plunger_tag.Component;
THole*& ramp_hole = control_ramp_hole_tag.Component;
TRamp*& ramp = control_ramp_tag.Component;
TWall*& rebo1 = control_rebo1_tag.Component;
TWall*& rebo2 = control_rebo2_tag.Component;
TWall*& rebo3 = control_rebo3_tag.Component;
TWall*& rebo4 = control_rebo4_tag.Component;
TRollover*& roll1 = control_roll1_tag.Component;
TRollover*& roll2 = control_roll2_tag.Component;
TRollover*& roll3 = control_roll3_tag.Component;
TRollover*& roll4 = control_roll4_tag.Component;
TRollover*& roll5 = control_roll5_tag.Component;
TRollover*& roll6 = control_roll6_tag.Component;
TRollover*& roll7 = control_roll7_tag.Component;
TRollover*& roll8 = control_roll8_tag.Component;
TLightRollover*& roll9 = control_roll9_tag.Component;
TRollover*& roll110 = control_roll110_tag.Component;
TRollover*& roll111 = control_roll111_tag.Component;
TRollover*& roll112 = control_roll112_tag.Component;
TRollover*& roll179 = control_roll179_tag.Component;
TRollover*& roll180 = control_roll180_tag.Component;
TRollover*& roll181 = control_roll181_tag.Component;
TRollover*& roll182 = control_roll182_tag.Component;
TRollover*& roll183 = control_roll183_tag.Component;
TRollover*& roll184 = control_roll184_tag.Component;
TSink*& sink1 = control_sink1_tag.Component;
TSink*& sink2 = control_sink2_tag.Component;
TSink*& sink3 = control_sink3_tag.Component;
TSink*& sink7 = control_sink7_tag.Component;
TSound*& soundwave3 = control_soundwave3_tag.Component;
TSound*& soundwave7 = control_soundwave7_tag.Component;
TSound*& soundwave8 = control_soundwave8_tag.Component;
TSound*& soundwave9 = control_soundwave9_tag.Component;
TSound*& soundwave10 = control_soundwave10_tag.Component;
TSound*& soundwave14_1 = control_soundwave14_1_tag.Component;
TSound*& soundwave14_2 = control_soundwave14_2_tag.Component;
TSound*& soundwave21 = control_soundwave21_tag.Component;
TSound*& soundwave23 = control_soundwave23_tag.Component;
TSound*& soundwave24 = control_soundwave24_tag.Component;
TSound*& soundwave25 = control_soundwave25_tag.Component;
TSound*& soundwave26 = control_soundwave26_tag.Component;
TSound*& soundwave27 = control_soundwave27_tag.Component;
TSound*& soundwave28 = control_soundwave28_tag.Component;
TSound*& soundwave30 = control_soundwave30_tag.Component;
TSound*& soundwave35_1 = control_soundwave35_1_tag.Component;
TSound*& soundwave35_2 = control_soundwave35_2_tag.Component;
TSound*& soundwave36_1 = control_soundwave36_1_tag.Component;
TSound*& soundwave36_2 = control_soundwave36_2_tag.Component;
TSound*& soundwave38 = control_soundwave38_tag.Component;
TSound*& soundwave39 = control_soundwave39_tag.Component;
TSound*& soundwave40 = control_soundwave40_tag.Component;
TSound*& soundwave41 = control_soundwave41_tag.Component;
TSound*& soundwave44 = control_soundwave44_tag.Component;
TSound*& soundwave45 = control_soundwave45_tag.Component;
TSound*& soundwave46 = control_soundwave46_tag.Component;
TSound*& soundwave47 = control_soundwave47_tag.Component;
TSound*& soundwave48 = control_soundwave48_tag.Component;
TSound*& soundwave49D = control_soundwave49D_tag.Component;
TSound*& soundwave50_1 = control_soundwave50_1_tag.Component;
TSound*& soundwave50_2 = control_soundwave50_2_tag.Component;
TSound*& soundwave52 = control_soundwave52_tag.Component;
TSound*& soundwave59 = control_soundwave59_tag.Component;
TPopupTarget*& target1 = control_target1_tag.Component;
TPopupTarget*& target2 = control_target2_tag.Component;
TPopupTarget*& target3 = control_target3_tag.Component;
TPopupTarget*& target4 = control_target4_tag.Component;
TPopupTarget*& target5 = control_target5_tag.Component;
TPopupTarget*& target6 = control_target6_tag.Component;
TPopupTarget*& target7 = control_target7_tag.Component;
TPopupTarget*& target8 = control_target8_tag.Component;
TPopupTarget*& target9 = control_target9_tag.Component;
TSoloTarget*& target10 = control_target10_tag.Component;
TSoloTarget*& target11 = control_target11_tag.Component;
TSoloTarget*& target12 = control_target12_tag.Component;
TSoloTarget*& target13 = control_target13_tag.Component;
TSoloTarget*& target14 = control_target14_tag.Component;
TSoloTarget*& target15 = control_target15_tag.Component;
TSoloTarget*& target16 = control_target16_tag.Component;
TSoloTarget*& target17 = control_target17_tag.Component;
TSoloTarget*& target18 = control_target18_tag.Component;
TSoloTarget*& target19 = control_target19_tag.Component;
TSoloTarget*& target20 = control_target20_tag.Component;
TSoloTarget*& target21 = control_target21_tag.Component;
TSoloTarget*& target22 = control_target22_tag.Component;
TTripwire*& trip1 = control_trip1_tag.Component;
TTripwire*& trip2 = control_trip2_tag.Component;
TTripwire*& trip3 = control_trip3_tag.Component;
TTripwire*& trip4 = control_trip4_tag.Component;
TTripwire*& trip5 = control_trip5_tag.Component;


TPinballTable* control::TableG;
component_info control::score_components[88]
{
	component_info{control_bump1_tag, {BumperControl, 4, control_bump_scores1}},
	component_info{control_bump2_tag, {BumperControl, 4, control_bump_scores1}},
	component_info{control_bump3_tag, {BumperControl, 4, control_bump_scores1}},
	component_info{control_bump4_tag, {BumperControl, 4, control_bump_scores1}},
	component_info{control_roll3_tag, {ReentryLanesRolloverControl, 1, control_roll_scores1}},
	component_info{control_roll2_tag, {ReentryLanesRolloverControl, 1, control_roll_scores1}},
	component_info{control_roll1_tag, {ReentryLanesRolloverControl, 1, control_roll_scores1}},
	component_info{control_attack_bump_tag, {BumperGroupControl, 0, nullptr}},
	component_info{control_bump5_tag, {BumperControl, 4, control_bump_scores2}},
	component_info{control_bump6_tag, {BumperControl, 4, control_bump_scores2}},
	component_info{control_bump7_tag, {BumperControl, 4, control_bump_scores2}},
	component_info{control_roll112_tag, {LaunchLanesRolloverControl, 1, control_roll_scores2}},
	component_info{control_roll111_tag, {LaunchLanesRolloverControl, 1, control_roll_scores2}},
	component_info{control_roll110_tag, {LaunchLanesRolloverControl, 1, control_roll_scores2}},
	component_info{control_launch_bump_tag, {BumperGroupControl, 0, nullptr}},
	component_info{control_rebo1_tag, {FlipperRebounderControl1, 1, control_rebo_score1}},
	component_info{control_rebo2_tag, {FlipperRebounderControl2, 1, control_rebo_score1}},
	component_info{control_rebo3_tag, {RebounderControl, 1, control_rebo_score1}},
	component_info{control_rebo4_tag, {RebounderControl, 1, control_rebo_score1}},
	component_info{control_kicker1_tag, {LeftKickerControl, 0, nullptr}},
	component_info{control_kicker2_tag, {RightKickerControl, 0, nullptr}},
	component_info{control_gate1_tag, {LeftKickerGateControl, 0, nullptr}},
	component_info{control_gate2_tag, {RightKickerGateControl, 0, nullptr}},
	component_info{control_oneway4_tag, {DeploymentChuteToEscapeChuteOneWayControl, 6, control_oneway4_score1}},
	component_info{control_oneway10_tag, {DeploymentChuteToTableOneWayControl, 0, nullptr}},
	component_info{control_block1_tag, {DrainBallBlockerControl, 0, nullptr}},
	component_info{control_ramp_tag, {LaunchRampControl, 1, control_ramp_score1}},
	component_info{control_ramp_hole_tag, {LaunchRampHoleControl, 0, nullptr}},
	component_info{control_roll4_tag, {OutLaneRolloverControl, 1, control_roll_score1}},
	component_info{control_roll8_tag, {OutLaneRolloverControl, 1, control_roll_score1}},
	component_info{control_lite17_tag, {ExtraBallLightControl, 0, nullptr}},
	component_info{control_roll6_tag, {ReturnLaneRolloverControl, 2, control_roll_score2}},
	component_info{control_roll7_tag, {ReturnLaneRolloverControl, 2, control_roll_score2}},
	component_info{control_roll5_tag, {BonusLaneRolloverControl, 1, control_roll_score3}},
	component_info{control_roll179_tag, {FuelRollover1Control, 1, control_roll_score4}},
	component_info{control_roll180_tag, {FuelRollover2Control, 1, control_roll_score4}},
	component_info{control_roll181_tag, {FuelRollover3Control, 1, control_roll_score4}},
	component_info{control_roll182_tag, {FuelRollover4Control, 1, control_roll_score4}},
	component_info{control_roll183_tag, {FuelRollover5Control, 1, control_roll_score4}},
	component_info{control_roll184_tag, {FuelRollover6Control, 1, control_roll_score4}},
	component_info{control_flag1_tag, {FlagControl, 2, control_flag_score1}},
	component_info{control_kickout2_tag, {HyperspaceKickOutControl, 5, control_kickout_score1}},
	component_info{control_hyper_lights_tag, {HyperspaceLightGroupControl, 0, nullptr}},
	component_info{control_flag2_tag, {FlagControl, 2, control_flag_score1}},
	component_info{control_sink1_tag, {WormHoleControl, 3, control_sink_score1}},
	component_info{control_sink2_tag, {WormHoleControl, 3, control_sink_score1}},
	component_info{control_sink3_tag, {WormHoleControl, 3, control_sink_score1}},
	component_info{control_flip1_tag, {LeftFlipperControl, 0, nullptr}},
	component_info{control_flip2_tag, {RightFlipperControl, 0, nullptr}},
	component_info{control_plunger_tag, {PlungerControl, 0, nullptr}},
	component_info{control_target1_tag, {BoosterTargetControl, 2, control_target_score1}},
	component_info{control_target2_tag, {BoosterTargetControl, 2, control_target_score1}},
	component_info{control_target3_tag, {BoosterTargetControl, 2, control_target_score1}},
	component_info{control_lite60_tag, {JackpotLightControl, 0, nullptr}},
	component_info{control_lite59_tag, {BonusLightControl, 0, nullptr}},
	component_info{control_target6_tag, {MedalTargetControl, 3, control_target_score2}},
	component_info{control_target5_tag, {MedalTargetControl, 3, control_target_score2}},
	component_info{control_target4_tag, {MedalTargetControl, 3, control_target_score2}},
	component_info{control_bumber_target_lights_tag, {MedalLightGroupControl, 0, nullptr}},
	component_info{control_target9_tag, {MultiplierTargetControl, 2, control_target_score3}},
	component_info{control_target8_tag, {MultiplierTargetControl, 2, control_target_score3}},
	component_info{control_target7_tag, {MultiplierTargetControl, 2, control_target_score3}},
	component_info{control_top_target_lights_tag, {MultiplierLightGroupControl, 0, nullptr}},
	component_info{control_target10_tag, {FuelSpotTargetControl, 1, control_target_score4}},
	component_info{control_target11_tag, {FuelSpotTargetControl, 1, control_target_score4}},
	component_info{control_target12_tag, {FuelSpotTargetControl, 1, control_target_score4}},
	component_info{control_target13_tag, {MissionSpotTargetControl, 1, control_target_score5}},
	component_info{control_target14_tag, {MissionSpotTargetControl, 1, control_target_score5}},
	component_info{control_target15_tag, {MissionSpotTargetControl, 1, control_target_score5}},
	component_info{control_target16_tag, {LeftHazardSpotTargetControl, 1, control_target_score6}},
	component_info{control_target17_tag, {LeftHazardSpotTargetControl, 1, control_target_score6}},
	component_info{control_target18_tag, {LeftHazardSpotTargetControl, 1, control_target_score6}},
	component_info{control_target19_tag, {RightHazardSpotTargetControl, 1, control_target_score6}},
	component_info{control_target20_tag, {RightHazardSpotTargetControl, 1, control_target_score6}},
	component_info{control_target21_tag, {RightHazardSpotTargetControl, 1, control_target_score6}},
	component_info{control_target22_tag, {WormHoleDestinationControl, 1, control_target_score7}},
	component_info{control_roll9_tag, {SpaceWarpRolloverControl, 1, control_roll_score5}},
	component_info{control_kickout3_tag, {BlackHoleKickoutControl, 1, control_kickout_score2}},
	component_info{control_kickout1_tag, {GravityWellKickoutControl, 1, control_kickout_score3}},
	component_info{control_drain_tag, {BallDrainControl, 0, nullptr}},
	component_info{control_oneway1_tag, {SkillShotGate1Control, 0, nullptr}},
	component_info{control_trip1_tag, {SkillShotGate2Control, 0, nullptr}},
	component_info{control_trip2_tag, {SkillShotGate3Control, 0, nullptr}},
	component_info{control_trip3_tag, {SkillShotGate4Control, 0, nullptr}},
	component_info{control_trip4_tag, {SkillShotGate5Control, 0, nullptr}},
	component_info{control_trip5_tag, {SkillShotGate6Control, 0, nullptr}},
	component_info{control_lite200_tag, {ShootAgainLightControl, 0, nullptr}},
	component_info{control_sink7_tag, {EscapeChuteSinkControl, 0, nullptr}},
};


component_tag_base* control::simple_components[145]
{
	&control_lite8_tag,
	&control_lite9_tag,
	&control_lite10_tag,
	&control_bmpr_inc_lights_tag,
	&control_lite171_tag,
	&control_lite170_tag,
	&control_lite169_tag,
	&control_ramp_bmpr_inc_lights_tag,
	&control_lite30_tag,
	&control_lite29_tag,
	&control_lite1_tag,
	&control_lite54_tag,
	&control_lite55_tag,
	&control_lite56_tag,
	&control_lite18_tag,
	&control_lite27_tag,
	&control_lite28_tag,
	&control_lite16_tag,
	&control_lite21_tag,
	&control_lite22_tag,
	&control_lite23_tag,
	&control_lite24_tag,
	&control_lite25_tag,
	&control_lite26_tag,
	&control_lite130_tag,
	&control_lite5_tag,
	&control_lite6_tag,
	&control_lite7_tag,
	&control_worm_hole_lights_tag,
	&control_lite4_tag,
	&control_lite2_tag,
	&control_lite3_tag,
	&control_bsink_arrow_lights_tag,
	&control_l_trek_lights_tag,
	&control_r_trek_lights_tag,
	&control_literoll179_tag,
	&control_literoll180_tag,
	&control_literoll181_tag,
	&control_literoll182_tag,
	&control_literoll183_tag,
	&control_literoll184_tag,
	&control_fuel_bargraph_tag,
	&control_lite20_tag,
	&control_lite19_tag,
	&control_lite61_tag,
	&control_lite58_tag,
	&control_lite11_tag,
	&control_lite12_tag,
	&control_lite13_tag,
	&control_lite70_tag,
	&control_lite71_tag,
	&control_lite72_tag,
	&control_top_circle_tgt_lights_tag,
	&control_lite101_tag,
	&control_lite102_tag,
	&control_lite103_tag,
	&control_ramp_tgt_lights_tag,
	&control_lite104_tag,
	&control_lite105_tag,
	&control_lite106_tag,
	&control_lite107_tag,
	&control_lite108_tag,
	&control_lite109_tag,
	&control_lchute_tgt_lights_tag,
	&control_bpr_solotgt_lights_tag,
	&control_lite110_tag,
	&control_lite62_tag,
	&control_lite67_tag,
	&control_lite68_tag,
	&control_lite69_tag,
	&control_lite131_tag,
	&control_lite132_tag,
	&control_lite133_tag,
	&control_skill_shot_lights_tag,
	&control_lite77_tag,
	&control_lite198_tag,
	&control_middle_circle_tag,
	&control_outer_circle_tag,
	&control_soundwave9_tag,
	&control_soundwave10_tag,
	&control_soundwave21_tag,
	&control_soundwave23_tag,
	&control_soundwave24_tag,
	&control_soundwave30_tag,
	&control_soundwave28_tag,
	&control_soundwave50_1_tag,
	&control_soundwave8_tag,
	&control_soundwave40_tag,
	&control_soundwave41_tag,
	&control_soundwave36_1_tag,
	&control_soundwave50_2_tag,
	&control_soundwave35_1_tag,
	&control_soundwave36_2_tag,
	&control_soundwave35_2_tag,
	&control_soundwave38_tag,
	&control_soundwave39_tag,
	&control_soundwave44_tag,
	&control_soundwave45_tag,
	&control_soundwave46_tag,
	&control_soundwave47_tag,
	&control_soundwave48_tag,
	&control_soundwave52_tag,
	&control_soundwave14_1_tag,
	&control_soundwave59_tag,
	&control_lite199_tag,
	&control_lite196_tag,
	&control_lite195_tag,
	&control_info_text_box_tag,
	&control_mission_text_box_tag,
	&control_soundwave27_tag,
	&control_lite84_tag,
	&control_lite85_tag,
	&control_soundwave14_2_tag,
	&control_soundwave3_tag,
	&control_soundwave26_tag,
	&control_soundwave49D_tag,
	&control_lite300_tag,
	&control_lite301_tag,
	&control_lite302_tag,
	&control_lite303_tag,
	&control_lite304_tag,
	&control_lite305_tag,
	&control_lite306_tag,
	&control_lite307_tag,
	&control_lite308_tag,
	&control_lite309_tag,
	&control_lite310_tag,
	&control_lite311_tag,
	&control_lite312_tag,
	&control_lite313_tag,
	&control_lite314_tag,
	&control_lite315_tag,
	&control_lite316_tag,
	&control_lite317_tag,
	&control_lite318_tag,
	&control_lite319_tag,
	&control_lite320_tag,
	&control_lite321_tag,
	&control_lite322_tag,
	&control_goal_lights_tag,
	&control_soundwave25_tag,
	&control_soundwave7_tag,
	&control_lite38_tag,
	&control_lite39_tag,
	&control_lite40_tag,
};

int control::waiting_deployment_flag;
bool control::table_unlimited_balls = false, control::easyMode = false;
int control::extraball_light_flag;
Msg control::RankRcArray[9] =
{
	Msg::STRING185,
	Msg::STRING186,
	Msg::STRING187,
	Msg::STRING188,
	Msg::STRING189,
	Msg::STRING190,
	Msg::STRING191,
	Msg::STRING192,
	Msg::STRING193
};
Msg control::MissionRcArray[17] =
{
	Msg::STRING161,
	Msg::STRING162,
	Msg::STRING163,
	Msg::STRING164,
	Msg::STRING165,
	Msg::STRING166,
	Msg::STRING167,
	Msg::STRING168,
	Msg::STRING169,
	Msg::STRING170,
	Msg::STRING171,
	Msg::STRING172,
	Msg::STRING173,
	Msg::STRING174,
	Msg::STRING175,
	Msg::STRING176,
	Msg::STRING177
};
int control::mission_select_scores[17] =
{
	10000,
	10000,
	10000,
	10000,
	20000,
	20000,
	20000,
	20000,
	20000,
	20000,
	20000,
	20000,
	20000,
	30000,
	30000,
	30000,
	30000
};

std::reference_wrapper<TSink*> control::WormholeSinkArray[3] =
{
	sink1, sink2, sink3
};

std::reference_wrapper<TLight*> control::WormholeLightArray1[3] =
{
	lite5, lite6, lite7
};

std::reference_wrapper<TLight*> control::WormholeLightArray2[3] =
{
	lite4, lite2, lite3
};


void control::make_links(TPinballTable* table)
{
	TableG = table;

	for (auto& score_component : score_components)
	{
		auto linkedComp = make_component_link(score_component.Tag);
		if (linkedComp)
		{
			linkedComp->Control = &score_component.Control;
		}
	}

	for (auto& simple_component : simple_components)
		make_component_link(*simple_component);
}

void control::ClearLinks()
{
	TableG = nullptr;
	for (auto& component : score_components)
		component.Tag.SetComponent(nullptr);
	for (auto& component : simple_components)
		component->SetComponent(nullptr);
}

TPinballComponent* control::make_component_link(component_tag_base& tag)
{
	if (tag.GetComponent())
		return tag.GetComponent();

	for (auto component: TableG->ComponentList)
	{
		if (component->GroupName)
		{
			if (!strcmp(component->GroupName, tag.Name))
			{
				tag.SetComponent(component);
				return component;
			}
		}
	}

	return nullptr;
}

void control::handler(MessageCode code, TPinballComponent* cmp)
{
	component_control* control = cmp->Control;
	
	if (control)
	{
		control->ControlFunc(code, cmp);
	}
	MissionControl(code, cmp);
}

void control::pbctrl_bdoor_controller(char key)
{
	// Buffer large enough for longest cheat + null
	static char cheatBuffer[11 + 1]{};
	static const char* bufferEnd = &cheatBuffer[11];
	static const char* quotes[8]
	{
		"Hey, is that a screen saver?",
		"I guess it has been a good week",
		"She may already be a glue bottle",
		"If you don't come in Saturday,\n...\n",
		"don't even bother coming in Sunday.",
		"Tomorrow already sucks",
		"I knew it worked too good to be right.",
		"World's most expensive flippers"
	};
	static const char* credits[35]
	{
		"Full Tilt! was created by Cinematronics",
		"for Maxis.",
		"Cinematronics Team:",
		"Programming\nMichael Sandige\nJohn Taylor",
		"Art\nJohn Frantz Jr.\nRyan Medeiros",
		"Design\nKevin Gliner",
		"Sound Effects\nMatt Ridgeway",
		"Donald S. Griffin",
		"Design Consultant\nMark Sprenger",
		"Music\nMatt Ridgeway",
		"Producer\nKevin Gliner",
		"Voices\nMike McGeary\nWilliam Rice",
		"Grand Poobah\nDavid Stafford",
		"Special Thanks\nPaula Sandige\nAlex St. John",
		"Brad Silverberg\nJeff Camp",
		"Danny Thorpe\nGreg Hospelhorn",
		"Maxis Team:",
		"Producer\nJohn Csicsery",
		"Product Manager\nLarry Lee",
		"Lead Tester\nMichael Gilmartin",
		"QA Manager\nAlan Barton",
		"Additional Testing\nJoe Longworth\nScott Shicoff",
		"Owen Nelson\nJohn \"Jussi\" Ylinen",
		"John Landes\nMarc Meyer",
		"Cathy Castro\nKeith Meyer",
		"Additional Art\nOcean Quigley",
		"Rick Macaraeg\nCharlie Aquilina",
		"Art Director\nSharon Barr",
		"Install Program\nKevin O'Hare",
		"Intro Music",
		"Brian Conrad",
		"John Csicsery",
		"Special Thanks\nSam Poole\nJoe Scirica",
		"Jeff Braun\nBob Derber\nAshley Csicsery",
		"Tom Forge\nWill \"Burr\" Wright",
	};

	// Original allowed to enter cheats only before the first launch.
	std::memmove(&cheatBuffer[0], &cheatBuffer[1], 10);
	cheatBuffer[10] = key;

	if (strcmp(bufferEnd - 11, "hidden test") == 0 ||
		strcmp(bufferEnd - 11, "hidden\ttest") == 0)
	{
		pb::cheat_mode ^= true;
	}
	else if (strcmp(bufferEnd - 4, "gmax") == 0)
	{
		GravityWellKickoutControl(MessageCode::ControlEnableMultiplier, nullptr);
	}
	else if (strcmp(bufferEnd - 4, "1max") == 0)
	{
		table_add_extra_ball(2.0);
	}
	else if (strcmp(bufferEnd - 4, "bmax") == 0)
	{
		table_unlimited_balls ^= true;
	}
	else if (strcmp(bufferEnd - 4, "rmax") == 0)
	{
		cheat_bump_rank();
	}
	else if (pb::FullTiltMode && strcmp(bufferEnd - 5, "quote") == 0)
	{
		// Developer easter egg type 'cheat' from Full Tilt
		float time = 0;
		for (auto quote : quotes)
			mission_text_box->Display(quote, time += 3, true);
		return;
	}
	else if (pb::FullTiltMode && strcmp(bufferEnd - 7, "credits") == 0)
	{
		// Full Tilt in-game credits, shown when idle for 60sec
		float time = 0;
		for (auto line : credits)
			mission_text_box->Display(line, time += 2, true);
		pb::CreditsActive = true;
		return;
	}
	else if (strcmp(bufferEnd - 9, "easy mode") == 0)
	{
		easyMode ^= true;
		if (easyMode) 
		{
			DrainBallBlockerControl(MessageCode::TBlockerEnable, block1);
			gate1->Message(MessageCode::TGateDisable, 0.0);
			gate2->Message(MessageCode::TGateDisable, 0.0);
		}
		else
			DrainBallBlockerControl(MessageCode::ControlTimerExpired, block1);
	}
	else
	{
		return;
	}

	TableG->CheatsUsed = 1;
}

void control::table_add_extra_ball(float count)
{
	++TableG->ExtraBalls;
	soundwave28->Play(nullptr, "table_add_extra_ball");
	info_text_box->Display(pb::get_rc_string(Msg::STRING110), count);
}

void control::table_set_bonus_hold()
{
	lite58->Message(MessageCode::TLightResetAndTurnOn, 0.0);
	info_text_box->Display(pb::get_rc_string(Msg::STRING153), 2.0);
}

void control::table_set_bonus()
{
	TableG->BonusScoreFlag = true;
	lite59->Message(MessageCode::TLightTurnOnTimed, 60.0);
	info_text_box->Display(pb::get_rc_string(Msg::STRING105), 2.0);
}

void control::table_set_jackpot()
{
	TableG->JackpotScoreFlag = true;
	lite60->Message(MessageCode::TLightTurnOnTimed, 60.0);
	info_text_box->Display(pb::get_rc_string(Msg::STRING116), 2.0);
}

void control::table_set_flag_lights()
{
	lite20->Message(MessageCode::TLightTurnOnTimed, 60.0);
	lite19->Message(MessageCode::TLightTurnOnTimed, 60.0);
	lite61->Message(MessageCode::TLightTurnOnTimed, 60.0);
	info_text_box->Display(pb::get_rc_string(Msg::STRING152), 2.0);
}

void control::table_set_multiball(float time)
{
	if (TableG->MultiballCount <= 1)
	{
		TableG->MultiballCount += 3;
		sink1->Message(MessageCode::TSinkResetTimer, time);
		sink2->Message(MessageCode::TSinkResetTimer, time);
		sink3->Message(MessageCode::TSinkResetTimer, time);
		lite38->Message(MessageCode::TLightFlasherStartTimed, -1.0f);
		lite39->Message(MessageCode::TLightFlasherStartTimed, -1.0f);
		lite40->Message(MessageCode::TLightFlasherStartTimed, -1.0f);
		info_text_box->Display(pb::get_rc_string(Msg::STRING117), 2.0);
		midi::play_track(MidiTracks::Track3, true);
	}
}

void control::table_bump_ball_sink_lock()
{
	if (TableG->MultiballCount <= 1)
	{
		TableG->MultiballCount--;
		if (TableG->BallLockedCounter == 2)
		{
			soundwave41->Play(nullptr, "table_bump_ball_sink_lock_set_multiball");
			table_set_multiball(2.0);
			TableG->BallLockedCounter = 0;
		}
		else
		{
			TableG->BallLockedCounter = TableG->BallLockedCounter + 1;
			soundwave44->Play(nullptr, "table_bump_ball_sink_lock");
			info_text_box->Display(pb::get_rc_string(Msg::STRING102), 2.0);
			TableG->Plunger->Message(MessageCode::PlungerRelaunchBall, 2.0f);
		}
	}
}

void control::table_set_replay(float value)
{
	lite199->Message(MessageCode::TLightResetAndTurnOn, 0.0);
	info_text_box->Display(pb::get_rc_string(Msg::STRING101), value);
}

void control::cheat_bump_rank()
{
	char Buffer[64]{};

	auto rank = middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0);
	if (rank < 9)
	{
		middle_circle->Message(MessageCode::TLightGroupResetAndTurnOn, 2.0f);
		auto rankText = pb::get_rc_string(RankRcArray[rank]);
		snprintf(Buffer,sizeof Buffer, pb::get_rc_string(Msg::STRING184), rankText);
		mission_text_box->Display(Buffer, 8.0);
		soundwave10->Play(nullptr, "cheat_bump_rank");
	}
}

void control::UnstuckBall(TBall& ball, int dt)
{
	if (!CheckBallInControlBounds(ball, *flip1) &&
		!CheckBallInControlBounds(ball, *flip2) &&
		!CheckBallInControlBounds(ball, *plunger))
	{
		if (ball.StuckCounter <= 20)
		{
			vector3 throwDir{0.0f, -1.0f, 0.0f};
			ball.throw_ball(&throwDir, 90.0f, 1.0f, 0.0f);
		}
		else
		{
			ball.Disable();
			TableG->MultiballCount--;
			plunger->Message(MessageCode::PlungerRelaunchBall, 0);
		}
	}
}

bool control::CheckBallInControlBounds(const TBall& ball, const TCollisionComponent& cmp)
{
	auto offset = TableG->CollisionCompOffset / 2.0f;
	return ball.ActiveFlag &&
		ball.Position.X >= cmp.AABB.XMin - offset &&
		ball.Position.X <= cmp.AABB.XMax + offset &&
		ball.Position.Y >= cmp.AABB.YMin - offset &&
		ball.Position.Y <= cmp.AABB.YMax + offset;
}

int control::SpecialAddScore(int score, bool mission)
{
	// FT: mission completion applies current jackpot
	if (mission && pb::FullTiltMode)
		score += TableG->JackpotScore;

	const auto bonus = TableG->BonusScoreFlag;
	const auto jackpot = TableG->JackpotScoreFlag;
	const auto scoreMult = TableG->ScoreMultiplier;

	TableG->BonusScoreFlag = false;
	TableG->JackpotScoreFlag = false;
	TableG->ScoreMultiplier = 0;
	const auto addedScore = TableG->AddScore(score);
	TableG->BonusScoreFlag = bonus;
	TableG->JackpotScoreFlag = jackpot;
	TableG->ScoreMultiplier = scoreMult;

	// FT: each mission starts with jackpot set to 5e5
	if (mission && pb::FullTiltMode)
		TableG->JackpotScore = 500000;

	return addedScore;
}

int control::AddRankProgress(int rank)
{
	char Buffer[64];
	int result = 0;

	lite16->Message(MessageCode::TLightResetAndTurnOn, 0.0);
	for (int index = rank; index; --index)
	{
		outer_circle->Message(MessageCode::TLightGroupResetAndTurnOn, 2.0);
	}

	int activeCount = outer_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0);
	int totalCount = outer_circle->Message(MessageCode::TLightGroupGetLightCount, 0.0);
	if (activeCount == totalCount)
	{
		result = 1;
		outer_circle->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 5.0);
		middle_circle->Message(MessageCode::TLightGroupReset, 0.0);
		int midActiveCount = middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0);
		if (midActiveCount < 9)
		{
			middle_circle->Message(MessageCode::TLightGroupResetAndTurnOn, 5.0);
			auto rankText = pb::get_rc_string(RankRcArray[midActiveCount]);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING184), rankText);
			mission_text_box->Display(Buffer, 8.0);
			soundwave10->Play(nullptr, "AddRankProgress");
		}
	}
	else if (activeCount >= 3 * totalCount / 4)
	{
		middle_circle->Message(MessageCode::TLightGroupAnimationForward, -1.0);
	}
	return result;
}

void control::AdvanceWormHoleDestination(int flag)
{
	int lite198Msg = lite198->MessageField;
	if (lite198Msg != 16 && lite198Msg != 22 && lite198Msg != 23)
	{
		int lite4Msg = lite4->MessageField;
		if (flag || lite4Msg)
		{
			int val1 = lite4Msg + 1;
			int val2 = val1;
			if (val1 == 4)
			{
				val1 = 1;
				val2 = 1;
			}
			bsink_arrow_lights->Message(MessageCode::TLightSetMessageField, static_cast<float>(val2));
			bsink_arrow_lights->Message(MessageCode::TLightSetOnStateBmpIndex, static_cast<float>(3 - val1));
			if (!lite4->light_on())
			{
				worm_hole_lights->Message(MessageCode::TLightResetAndTurnOn, 0.0);
				bsink_arrow_lights->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			}
		}
	}
}

void control::FlipperRebounderControl1(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		lite84->Message(MessageCode::TLightTurnOnTimed, 0.1f);
		auto score = caller->get_scoring(0);
		TableG->AddScore(score);
	}
}

void control::FlipperRebounderControl2(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		lite85->Message(MessageCode::TLightTurnOnTimed, 0.1f);
		int score = caller->get_scoring(0);
		TableG->AddScore(score);
	}
}

void control::RebounderControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::BumperControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		TableG->AddScore(caller->get_scoring(static_cast<TBumper*>(caller)->BmpIndex));
	}
}

void control::LeftKickerControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlTimerExpired && !easyMode)
		gate1->Message(MessageCode::TGateEnable, 0.0);
}

void control::RightKickerControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlTimerExpired && !easyMode)
		gate2->Message(MessageCode::TGateEnable, 0.0);
}

void control::LeftKickerGateControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::TGateDisable)
	{
		lite30->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 5.0);
		lite196->Message(MessageCode::TLightFlasherStartTimed, 5.0);
	}
	else if (code == MessageCode::TGateEnable)
	{
		lite30->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite196->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	}
}

void control::RightKickerGateControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::TGateDisable)
	{
		lite29->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 5.0);
		lite195->Message(MessageCode::TLightFlasherStartTimed, 5.0);
	}
	else if (code == MessageCode::TGateEnable)
	{
		lite29->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite195->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	}
}

void control::DeploymentChuteToEscapeChuteOneWayControl(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];
	if (code == MessageCode::ControlCollision)
	{
		int count = skill_shot_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0);
		if (count)
		{
			soundwave3->Play(nullptr, "DeploymentChuteToEscapeChuteOneWayControl");
			int score = TableG->AddScore(caller->get_scoring(count - 1));
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING122), score);
			info_text_box->Display(Buffer, 2.0);
			if (!lite56->light_on())
			{
				l_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
				l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				r_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
				r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			}
			skill_shot_lights->Message(MessageCode::TLightGroupFlashWhenOn, 1.0);
		}
	}
}

void control::DeploymentChuteToTableOneWayControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
		skill_shot_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
}

void control::DrainBallBlockerControl(MessageCode code, TPinballComponent* caller)
{
	auto block = static_cast<TBlocker*>(caller);
	switch (code)
	{
	case MessageCode::TBlockerEnable:
		{
			block->MessageField = 1;
			auto blockerDuration = !easyMode ? static_cast<float>(block->InitialDuration) : -1;
			block->Message(MessageCode::TBlockerEnable, blockerDuration);
			lite1->Message(MessageCode::TLightTurnOnTimed, blockerDuration);
			break;
		}
	case MessageCode::ControlTimerExpired:
		{
			if (block->MessageField == 1)
			{
				block->MessageField = 2;
				auto blockerDuration = static_cast<float>(block->ExtendedDuration);
				block->Message(MessageCode::TBlockerRestartTimeout, blockerDuration);
				lite1->Message(MessageCode::TLightFlasherStartTimed, blockerDuration);
				break;
			}
			else
			{
				block->MessageField = 0;
				block->Message(MessageCode::TBlockerDisable, 0.0);
				break;
			}
		}
	default: break;
	}
}

void control::LaunchRampControl(MessageCode code, TPinballComponent* caller)
{
	TSound* sound;
	char Buffer[64];

	if (code == MessageCode::ControlCollision)
	{
		int someFlag = 0;
		if (lite54->light_on())
		{
			someFlag = 1;
			int addedScore = SpecialAddScore(TableG->ReflexShotScore);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING111), addedScore);
			info_text_box->Display(Buffer, 2.0);
		}
		if (lite55->light_on())
			someFlag |= 2u;
		if (lite56->light_on())
			someFlag |= 4u;
		if (someFlag)
		{
			if (someFlag == 1)
			{
				sound = soundwave21;
			}
			else if (someFlag < 1 || someFlag > 3)
			{
				sound = soundwave24;
			}
			else
			{
				sound = soundwave23;
			}
		}
		else
		{
			TableG->AddScore(caller->get_scoring(0));
			sound = soundwave30;
		}
		sound->Play(lite198, "LaunchRampControl");
	}
}

void control::LaunchRampHoleControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlBallReleased)
		lite54->Message(MessageCode::TLightFlasherStartTimed, 5.0);
}

void control::SpaceWarpRolloverControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		lite27->Message(MessageCode::TLightResetAndTurnOn, 0.0);
		lite28->Message(MessageCode::TLightResetAndTurnOn, 0.0);
	}
}

void control::ReentryLanesRolloverControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (!lite56->light_on() && l_trek_lights->Message(MessageCode::TLightGroupGetMessage2, 0.0))
		{
			l_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
			l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			r_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
			r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		}

		TLight* light;
		if (roll3 == caller)
		{
			light = lite8;
		}
		else if (roll2 == caller)
		{
			light = lite9;
		}
		else
		{
			light = lite10;
		}
		if (!light->FlasherOnFlag)
		{
			if (light->LightOnFlag)
			{
				if (!pb::FullTiltMode)
				{
					light->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				}
			}
			else
			{
				light->Message(MessageCode::TLightResetAndTurnOn, 0.0);
				int activeCount = bmpr_inc_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0);
				if (activeCount == bmpr_inc_lights->Message(MessageCode::TLightGroupGetLightCount, 0.0))
				{
					bmpr_inc_lights->Message(MessageCode::TLightFlasherStartTimed, 5.0);
					bmpr_inc_lights->Message(MessageCode::TLightTurnOff, 0.0);
					if (bump1->BmpIndex < 3)
					{
						attack_bump->Message(MessageCode::TBumperIncBmpIndex, 0.0);
						info_text_box->Display(pb::get_rc_string(Msg::STRING106), 2.0);
					}
					attack_bump->Message(MessageCode::TComponentGroupResetNotifyTimer, 60.0);
				}
			}
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::BumperGroupControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlNotifyTimerExpired)
	{
		caller->Message(MessageCode::TComponentGroupResetNotifyTimer, 60.0);
		caller->Message(MessageCode::TBumperDecBmpIndex, 0.0);
	}
}

void control::LaunchLanesRolloverControl(MessageCode code, TPinballComponent* caller)
{
	TLight* light;

	if (code == MessageCode::ControlCollision)
	{
		if (roll112 == caller)
		{
			light = lite171;
		}
		else
		{
			light = lite170;
			if (roll111 != caller)
				light = lite169;
		}
		if (!light->FlasherOnFlag)
		{
			if (light->LightOnFlag)
			{
				if (!pb::FullTiltMode)
				{
					light->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				}
			}
			else
			{
				light->Message(MessageCode::TLightResetAndTurnOn, 0.0);
				int msg1 = ramp_bmpr_inc_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0);
				if (msg1 == ramp_bmpr_inc_lights->Message(MessageCode::TLightGroupGetLightCount, 0.0))
				{
					ramp_bmpr_inc_lights->Message(MessageCode::TLightFlasherStartTimed, 5.0);
					ramp_bmpr_inc_lights->Message(MessageCode::TLightTurnOff, 0.0);
					if (bump5->BmpIndex < 3)
					{
						launch_bump->Message(MessageCode::TBumperIncBmpIndex, 0.0);
						info_text_box->Display(pb::get_rc_string(Msg::STRING107), 2.0);
					}
					launch_bump->Message(MessageCode::TComponentGroupResetNotifyTimer, 60.0);
				}
			}
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::OutLaneRolloverControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (lite17->light_on() || lite18->light_on())
		{
			table_add_extra_ball(2.0);
			lite17->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite18->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		}
		else
		{
			soundwave26->Play(caller, "OutLaneRolloverControl");
		}
		if (roll4 == caller)
		{
			if (lite30->light_on())
			{
				lite30->Message(MessageCode::TLightFlasherStart, 0.0);
				lite196->Message(MessageCode::TLightFlasherStart, 0.0);
			}
		}
		else if (lite29->light_on())
		{
			lite29->Message(MessageCode::TLightFlasherStart, 0.0);
			lite195->Message(MessageCode::TLightFlasherStart, 0.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::ExtraBallLightControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::TLightResetAndTurnOn)
	{
		lite17->Message(MessageCode::TLightTurnOnTimed, 55.0);
		lite18->Message(MessageCode::TLightTurnOnTimed, 55.0);
		extraball_light_flag = 1;
	}
	else if (code == MessageCode::ControlTimerExpired)
	{
		if (extraball_light_flag)
		{
			lite17->Message(MessageCode::TLightFlasherStartTimed, 5.0);
			lite18->Message(MessageCode::TLightFlasherStartTimed, 5.0);
			extraball_light_flag = 0;
		}
	}
}

void control::ReturnLaneRolloverControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (roll6 == caller)
		{
			if (lite27->light_on())
			{
				lite59->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite27->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				TableG->AddScore(caller->get_scoring(1));
			}
			else
				TableG->AddScore(caller->get_scoring(0));
		}
		else if (roll7 == caller)
		{
			if (lite28->light_on())
			{
				lite59->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite28->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				TableG->AddScore(caller->get_scoring(1));
			}
			else
				TableG->AddScore(caller->get_scoring(0));
		}
	}
}

void control::BonusLaneRolloverControl(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code == MessageCode::ControlCollision)
	{
		if (lite16->light_on())
		{
			int addedScore = SpecialAddScore(TableG->BonusScore);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING104), addedScore);
			info_text_box->Display(Buffer, 2.0);
			lite16->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			soundwave50_1->Play(caller, "BonusLaneRolloverControl1");
		}
		else
		{
			TableG->AddScore(caller->get_scoring(0));
			soundwave25->Play(caller, "BonusLaneRolloverControl2");
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 11.0);
	}
}

void control::FuelRollover1Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 1)
		{
			literoll179->Message(MessageCode::TLightTurnOffTimed, 0.05f);
		}
		else
		{
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 1.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::FuelRollover2Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 3)
		{
			literoll180->Message(MessageCode::TLightTurnOffTimed, 0.05f);
		}
		else
		{
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 3.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::FuelRollover3Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 5)
		{
			literoll181->Message(MessageCode::TLightTurnOffTimed, 0.05f);
		}
		else
		{
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 5.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::FuelRollover4Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 7)
		{
			literoll182->Message(MessageCode::TLightTurnOffTimed, 0.05f);
		}
		else
		{
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 7.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::FuelRollover5Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 9)
		{
			literoll183->Message(MessageCode::TLightTurnOffTimed, 0.05f);
		}
		else
		{
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 9.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::FuelRollover6Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 11)
		{
			literoll184->Message(MessageCode::TLightTurnOffTimed, 0.05f);
		}
		else
		{
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 11.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::HyperspaceLightGroupControl(MessageCode code, TPinballComponent* caller)
{
	switch (code)
	{
	case MessageCode::TLightGroupNull:
		caller->Message(MessageCode::TLightTurnOff, 0.0);
		break;
	case MessageCode::TLightGroupResetAndTurnOn:
		caller->Message(MessageCode::TLightGroupResetAndTurnOn, 2.0);
		caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 60.0);
		break;
	case MessageCode::ControlNotifyTimerExpired:
		caller->Message(MessageCode::TLightGroupOffsetAnimationBackward, 0.0);
		if (caller->Message(MessageCode::TLightGroupGetOnCount, 0.0))
			caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 60.0);
		break;
	default: break;
	}
}

void control::WormHoleControl(MessageCode code, TPinballComponent* caller)
{
	int sinkFlag2;
	TSink* sink = static_cast<TSink*>(caller);

	if (code == MessageCode::ControlCollision)
	{
		int sinkFlag = 0;
		if (sink1 != sink)
		{
			sinkFlag = sink2 != sink;
			++sinkFlag;
		}

		int lite4Msg = lite4->MessageField;
		if (lite4Msg)
		{
			lite4->MessageField = 0;
			worm_hole_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			bsink_arrow_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite110->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (lite4Msg == sinkFlag + 1)
			{
				if (TableG->MultiballFlag)
				{
					if (TableG->MultiballCount == 1)
					{
						table_bump_ball_sink_lock();
						TableG->AddScore(10000);
						return;
					}
					else
					{
						table_set_replay(4.0);
						TableG->AddScore(50000);
					}
				}
				else
				{
					
					table_set_replay(4.0);
					TableG->AddScore(sink->get_scoring(1));
				}

				info_text_box->Display(pb::get_rc_string(Msg::STRING150), 2.0);
				WormholeLightArray1[sinkFlag].get()->Message(MessageCode::TLightFlasherStartTimedThenStayOff, sink->TimerTime);
				WormholeLightArray2[sinkFlag].get()->Message(MessageCode::TLightSetOnStateBmpIndex, static_cast<float>(2 - sinkFlag));
				WormholeLightArray2[sinkFlag].get()->Message(MessageCode::TLightFlasherStartTimedThenStayOff, sink->TimerTime);
				WormholeSinkArray[sinkFlag].get()->Message(MessageCode::TSinkResetTimer, sink->TimerTime);
				return;
			}
			TableG->AddScore(sink->get_scoring(2));
			sinkFlag2 = lite4Msg - 1;
		}
		else
		{
			TableG->AddScore(sink->get_scoring(0));
			sinkFlag2 = sinkFlag;
		}

		WormholeLightArray1[sinkFlag2].get()->Message(MessageCode::TLightFlasherStartTimedThenStayOff, sink->TimerTime);
		WormholeLightArray2[sinkFlag2].get()->Message(MessageCode::TLightSetOnStateBmpIndex, static_cast<float>(2 - sinkFlag2));
		WormholeLightArray2[sinkFlag2].get()->Message(MessageCode::TLightFlasherStartTimedThenStayOff, sink->TimerTime);
		WormholeSinkArray[sinkFlag2].get()->Message(MessageCode::TSinkResetTimer, sink->TimerTime);
		info_text_box->Display(pb::get_rc_string(Msg::STRING150), 2.0);
	}
}

void control::LeftFlipperControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::TLightTurnOn)
	{
		bmpr_inc_lights->Message(MessageCode::TLightGroupStepBackward, 0.0);
		ramp_bmpr_inc_lights->Message(MessageCode::TLightGroupStepBackward, 0.0);
	}
}

void control::RightFlipperControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::TLightTurnOn)
	{
		bmpr_inc_lights->Message(MessageCode::TLightGroupStepForward, 0.0);
		ramp_bmpr_inc_lights->Message(MessageCode::TLightGroupStepForward, 0.0);
	}
}

void control::JackpotLightControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlTimerExpired)
		TableG->JackpotScoreFlag = false;
}

void control::BonusLightControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlTimerExpired)
		TableG->BonusScoreFlag = false;
}

void control::BoosterTargetControl(MessageCode code, TPinballComponent* caller)
{
	TSound* sound = nullptr;

	if (code == MessageCode::ControlCollision && !caller->MessageField)
	{
		caller->MessageField = 1;
		if (target1->MessageField
			+ target2->MessageField
			+ target3->MessageField != 3)
		{
			TableG->AddScore(caller->get_scoring(0));
			return;
		}
		if (lite61->light_on())
		{
			if (lite60->light_on())
			{
				if (lite59->light_on())
				{
					if (lite58->light_on())
					{
						TableG->AddScore(caller->get_scoring(1));
					}
					else
					{
						table_set_bonus_hold();
					}
					sound = soundwave48;
				}
				else
				{
					table_set_bonus();
					sound = soundwave46;
				}
			}
			else
			{
				table_set_jackpot();
				sound = soundwave45;
			}
		}
		else
		{
			int msg = lite198->MessageField;
			if (msg != 15 && msg != 29)
			{
				table_set_flag_lights();
				sound = soundwave47;
			}
		}
		if (sound)
			sound->Play(caller, "BoosterTargetControl");

		target1->MessageField = 0;
		target1->Message(MessageCode::TPopupTargetEnable, 0.0);
		target2->MessageField = 0;
		target2->Message(MessageCode::TPopupTargetEnable, 0.0);
		target3->MessageField = 0;
		target3->Message(MessageCode::TPopupTargetEnable, 0.0);
		TableG->AddScore(caller->get_scoring(1));
	}
}

void control::MedalLightGroupControl(MessageCode code, TPinballComponent* caller)
{
	switch (code)
	{
	case MessageCode::TLightGroupNull:
		caller->Message(MessageCode::TLightTurnOff, 0.0);
		break;
	case MessageCode::TLightGroupResetAndTurnOn:
		caller->Message(MessageCode::TLightGroupResetAndTurnOn, 2.0);
		caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 30.0);
		break;
	case MessageCode::ControlNotifyTimerExpired:
		caller->Message(MessageCode::TLightGroupOffsetAnimationBackward, 0.0);
		if (caller->Message(MessageCode::TLightGroupGetOnCount, 0.0))
			caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 30.0);
		break;
	default: break;
	}
}

void control::MultiplierLightGroupControl(MessageCode code, TPinballComponent* caller)
{
	switch (code)
	{
	case MessageCode::TLightGroupNull:
		caller->Message(MessageCode::TLightTurnOff, 0.0);
		break;
	case MessageCode::TLightGroupResetAndTurnOn:
		caller->Message(MessageCode::TLightGroupResetAndTurnOn, 2.0);
		caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 30.0);
		break;
	case MessageCode::ControlNotifyTimerExpired:
		if (TableG->ScoreMultiplier)
			TableG->ScoreMultiplier = TableG->ScoreMultiplier - 1;
		caller->Message(MessageCode::TLightGroupOffsetAnimationBackward, 0.0);
		if (caller->Message(MessageCode::TLightGroupGetOnCount, 0.0))
			caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 30.0);
		break;
	case MessageCode::ControlEnableMultiplier:
		TableG->ScoreMultiplier = 4;
		caller->Message(MessageCode::TLightResetAndTurnOn, 0.0);
		caller->Message(MessageCode::TLightGroupRestartNotifyTimer, 30.0);
		info_text_box->Display(pb::get_rc_string(Msg::STRING160), 2.0);
		break;
	case MessageCode::ControlDisableMultiplier:
		TableG->ScoreMultiplier = 0;
		caller->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		caller->Message(MessageCode::TLightGroupRestartNotifyTimer, -1.0);
		break;
	default:
		break;
	}
}

void control::FuelSpotTargetControl(MessageCode code, TPinballComponent* caller)
{
	TPinballComponent* liteComp;

	if (code == MessageCode::ControlCollision)
	{
		if (target10 == caller)
		{
			liteComp = lite70;
		}
		else
		{
			liteComp = lite71;
			if (target11 != caller)
				liteComp = lite72;
		}
		liteComp->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 2.0);
		TableG->AddScore(caller->get_scoring(0));
		if (top_circle_tgt_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0) == 3)
		{
			top_circle_tgt_lights->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 2.0);
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 11.0);
			soundwave25->Play(caller, "FuelSpotTargetControl1");
			info_text_box->Display(pb::get_rc_string(Msg::STRING145), 2.0);
		}
		else
		{
			soundwave49D->Play(caller, "FuelSpotTargetControl2");
		}
	}
}

void control::MissionSpotTargetControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		TPinballComponent* lite;
		if (target13 == caller)
		{
			lite101->MessageField |= 1u;
			lite = lite101;
		}
		else if (target14 == caller)
		{
			lite101->MessageField |= 2u;
			lite = lite102;
		}
		else
		{
			lite101->MessageField |= 4u;
			lite = lite103;
		}
		lite->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 2.0);

		TSound* sound;
		if (!lite198->light_on() || lite198->FlasherOnFlag)
		{
			sound = soundwave52;
		}
		else
			sound = soundwave49D;
		sound->Play(caller, "MissionSpotTargetControl");
		TableG->AddScore(caller->get_scoring(0));
		if (ramp_tgt_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0) == 3)
			ramp_tgt_lights->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 2.0);
	}
}

void control::LeftHazardSpotTargetControl(MessageCode code, TPinballComponent* caller)
{
	TPinballComponent* lite;

	if (code == MessageCode::ControlCollision)
	{
		if (target16 == caller)
		{
			lite104->MessageField |= 1u;
			lite = lite104;
		}
		else if (target17 == caller)
		{
			lite104->MessageField |= 2u;
			lite = lite105;
		}
		else
		{
			lite104->MessageField |= 4u;
			lite = lite106;
		}
		lite->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 2.0);
		TableG->AddScore(caller->get_scoring(0));
		if (lchute_tgt_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0) == 3)
		{
			soundwave14_1->Play(caller, "LeftHazardSpotTargetControl1");
			gate1->Message(MessageCode::TGateDisable, 0.0);
			lchute_tgt_lights->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 2.0);
		}
		else
		{
			soundwave49D->Play(caller, "LeftHazardSpotTargetControl2");
		}
	}
}

void control::RightHazardSpotTargetControl(MessageCode code, TPinballComponent* caller)
{
	TPinballComponent* light;

	if (code == MessageCode::ControlCollision)
	{
		if (target19 == caller)
		{
			lite107->MessageField |= 1u;
			light = lite107;
		}
		else if (target20 == caller)
		{
			lite107->MessageField |= 2u;
			light = lite108;
		}
		else
		{
			lite107->MessageField |= 4u;
			light = lite109;
		}
		light->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 2.0);
		TableG->AddScore(caller->get_scoring(0));
		if (bpr_solotgt_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0) == 3)
		{
			soundwave14_1->Play(caller, "RightHazardSpotTargetControl1");
			gate2->Message(MessageCode::TGateDisable, 0.0);
			bpr_solotgt_lights->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 2.0);
		}
		else
		{
			soundwave49D->Play(caller, "RightHazardSpotTargetControl2");
		}
	}
}

void control::WormHoleDestinationControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (!lite110->light_on())
		{
			lite110->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 3.0);
			info_text_box->Display(pb::get_rc_string(Msg::STRING194), 2.0);
		}
		TableG->AddScore(caller->get_scoring(0));
		AdvanceWormHoleDestination(1);
	}
}

void control::BlackHoleKickoutControl(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code == MessageCode::ControlCollision)
	{
		int addedScore = TableG->AddScore(caller->get_scoring(0));
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING181), addedScore);
		info_text_box->Display(Buffer, 2.0);
		caller->Message(MessageCode::TKickoutRestartTimer, -1.0);
	}
}

void control::FlagControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlSpinnerLoopReset)
	{
		AdvanceWormHoleDestination(0);
	}
	else if (code == MessageCode::ControlCollision)
	{
		int score = caller->get_scoring(lite20->light_on());
		TableG->AddScore(score);
	}
}

void control::GravityWellKickoutControl(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	switch (code)
	{
	case MessageCode::ControlCollision:
		{
			auto addedScore = TableG->AddScore(caller->get_scoring(0));
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING182), addedScore);
			info_text_box->Display(Buffer, 2.0);
			lite62->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			caller->ActiveFlag = 0;
			auto duration = soundwave7->Play(lite62, "GravityWellKickoutControl");
			caller->Message(MessageCode::TKickoutRestartTimer, duration);
			break;
		}
	case MessageCode::ControlEnableMultiplier:
		{
			auto score = reinterpret_cast<size_t>(caller);
			if (score)
			{
				snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING183), score);
			}
			else
			{
				snprintf(Buffer, sizeof Buffer, "%s", pb::get_rc_string(Msg::STRING146));
			}
			info_text_box->Display(Buffer, 2.0);
			lite62->Message(MessageCode::TLightFlasherStart, 0.0);
			kickout1->ActiveFlag = 1;
			break;
		}
	case MessageCode::Reset:
		kickout1->ActiveFlag = 0;
		break;
	default: break;
	}
}

void control::SkillShotGate1Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		lite200->Message(MessageCode::TLightTurnOnTimed, 5.0);
		if (lite67->light_on())
		{
			skill_shot_lights->Message(MessageCode::TLightGroupReset, 0.0);
			skill_shot_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite67->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite54->Message(MessageCode::TLightFlasherStartTimed, 5.0);
			lite25->Message(MessageCode::TLightFlasherStartTimed, 5.0);
			fuel_bargraph->Message(MessageCode::TLightGroupToggleSplitIndex, 11.0);
			soundwave14_2->Play(lite67, "SkillShotGate1Control");
		}
	}
}

void control::SkillShotGate2Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (lite67->light_on())
		{
			lite68->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			soundwave14_2->Play(lite68, "SkillShotGate2Control");
		}
	}
}

void control::SkillShotGate3Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (lite67->light_on())
		{
			lite69->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			soundwave14_2->Play(lite69, "SkillShotGate3Control");
		}
	}
}

void control::SkillShotGate4Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (lite67->light_on())
		{
			lite131->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			soundwave14_2->Play(lite131, "SkillShotGate4Control");
		}
	}
}

void control::SkillShotGate5Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (lite67->light_on())
		{
			lite132->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			soundwave14_2->Play(lite132, "SkillShotGate5Control");
		}
	}
}

void control::SkillShotGate6Control(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		if (lite67->light_on())
		{
			lite133->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			soundwave14_2->Play(lite133, "SkillShotGate6Control");
		}
	}
}

void control::ShootAgainLightControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlTimerExpired)
	{
		if (caller->MessageField)
		{
			caller->MessageField = 0;
		}
		else
		{
			caller->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 5.0);
			caller->MessageField = 1;
		}
	}
}

void control::EscapeChuteSinkControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision)
	{
		caller->Message(MessageCode::TSinkResetTimer, -1.0f);
	}
}

void control::MissionControl(MessageCode code, TPinballComponent* caller)
{
	if (!lite198)
		return;

	int lite198Msg = lite198->MessageField;
	switch (code)
	{
	case MessageCode::TLightGroupCountdownEnded:
		if (fuel_bargraph == caller && lite198Msg > 1)
		{
			l_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
			l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			r_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
			r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING210), 4.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
		break;
	case MessageCode::ControlTimerExpired:
		if (fuel_bargraph == caller && lite198Msg)
		{
			if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0) == 1)
			{
				mission_text_box->Display(pb::get_rc_string(Msg::STRING217), 4.0);
			}
			break;
		}
		if (mission_text_box == caller)
			code = MessageCode::ControlMissionStarted;
		break;
	case MessageCode::Resume:
		code = MessageCode::ControlMissionStarted;
		break;
	default:
		break;
	}

	switch (lite198Msg)
	{
	case 0:
		WaitingDeploymentController(code, caller);
		break;
	case 1:
		SelectMissionController(code, caller);
		break;
	case 2:
		PracticeMissionController(code, caller);
		break;
	case 3:
		LaunchTrainingController(code, caller);
		break;
	case 4:
		ReentryTrainingController(code, caller);
		break;
	case 5:
		ScienceMissionController(code, caller);
		break;
	case 6:
		StrayCometController(code, caller);
		break;
	case 7:
		BlackHoleThreatController(code, caller);
		break;
	case 8:
		SpaceRadiationController(code, caller);
		break;
	case 9:
		BugHuntController(code, caller);
		break;
	case 10:
		AlienMenaceController(code, caller);
		break;
	case 11:
		RescueMissionController(code, caller);
		break;
	case 12:
		SatelliteController(code, caller);
		break;
	case 13:
		ReconnaissanceController(code, caller);
		break;
	case 14:
		DoomsdayMachineController(code, caller);
		break;
	case 15:
		CosmicPlagueController(code, caller);
		break;
	case 16:
		SecretMissionYellowController(code, caller);
		break;
	case 17:
		TimeWarpController(code, caller);
		break;
	case 18:
		MaelstromController(code, caller);
		break;
	/*case 19: QuoteController*/
	case 20:
		AlienMenacePartTwoController(code, caller);
		break;
	case 21:
		CosmicPlaguePartTwoController(code, caller);
		break;
	case 22:
		SecretMissionRedController(code, caller);
		break;
	case 23:
		SecretMissionGreenController(code, caller);
		break;
	case 24:
		TimeWarpPartTwoController(code, caller);
		break;
	case 25:
		MaelstromPartTwoController(code, caller);
		break;
	case 26:
		MaelstromPartThreeController(code, caller);
		break;
	case 27:
		MaelstromPartFourController(code, caller);
		break;
	case 28:
		MaelstromPartFiveController(code, caller);
		break;
	case 29:
		MaelstromPartSixController(code, caller);
		break;
	case 30:
		MaelstromPartSevenController(code, caller);
		break;
	case 31:
		MaelstromPartEightController(code, caller);
		break;
	case 32:
		GameoverController(code, caller);
		break;
	default:
		UnselectMissionController(code, caller);
		break;
	}
}

void control::HyperspaceKickOutControl(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
		return;

	auto activeCount = hyper_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0);
	HyperspaceLightGroupControl(MessageCode::TLightGroupResetAndTurnOn, hyper_lights);
	switch (activeCount)
	{
	case 0:
		{
			auto addedScore = TableG->AddScore(caller->get_scoring(0));
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING113), addedScore);
			info_text_box->Display(Buffer, 2.0);
			break;
		}
	case 1:
		{
			if (!pb::FullTiltMode)
			{
				auto addedScore = SpecialAddScore(TableG->JackpotScore);
				snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING115), addedScore);
				info_text_box->Display(Buffer, 2.0);
				TableG->JackpotScore = 20000;
			}
			else
			{
				TableG->JackpotScore *= 2;
				if (TableG->JackpotScore > 10000000)
					TableG->JackpotScore = 10000000;
				info_text_box->Display(pb::get_rc_string(Msg::ControlJackpotDoubled), 2.0);
			}
			break;
		}
	case 2:
		{
			DrainBallBlockerControl(MessageCode::TBlockerEnable, block1);
			auto addedScore = TableG->AddScore(caller->get_scoring(2));
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING103), addedScore);
			info_text_box->Display(Buffer, 2.0);
			break;
		}
	case 3:
		{
			ExtraBallLightControl(MessageCode::TLightResetAndTurnOn, nullptr);
			auto addedScore = TableG->AddScore(caller->get_scoring(3));
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING109), addedScore);
			info_text_box->Display(Buffer, 2.0);
			break;
		}
	case 4:
		{
			hyper_lights->Message(MessageCode::TLightTurnOff, 0.0);
			size_t addedScore = TableG->AddScore(caller->get_scoring(4));
			GravityWellKickoutControl(MessageCode::ControlEnableMultiplier, reinterpret_cast<TPinballComponent*>(addedScore));
			break;
		}
	default:
		break;
	}

	int someFlag = 0;
	if (lite25->light_on())
	{
		someFlag = 1;
		auto addedScore = SpecialAddScore(TableG->ReflexShotScore);
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING111), addedScore);
		info_text_box->Display(Buffer, 2.0);
	}
	if (lite26->light_on())
		someFlag |= 2u;
	if (lite130->light_on())
	{
		someFlag |= 4u;
		lite130->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		MultiplierLightGroupControl(MessageCode::ControlEnableMultiplier, top_target_lights);
		bumber_target_lights->Message(MessageCode::TLightResetAndTurnOn, 0.0);
		table_set_jackpot();
		table_set_bonus();
		table_set_flag_lights();
		table_set_bonus_hold();
		lite27->Message(MessageCode::TLightResetAndTurnOn, 0.0);
		lite28->Message(MessageCode::TLightResetAndTurnOn, 0.0);
		ExtraBallLightControl(MessageCode::TLightResetAndTurnOn, nullptr);
		DrainBallBlockerControl(MessageCode::TBlockerEnable, block1);

		if (TableG->MultiballFlag)
		{
			auto duration = soundwave41->Play(nullptr, "HyperspaceKickOutControl_setMultiball");
			table_set_multiball(duration);
		}
		if (TableG->JackpotScore < 100000)
			TableG->JackpotScore = 100000;
		if (TableG->BonusScore < 100000)
			TableG->BonusScore = 100000;
		GravityWellKickoutControl(MessageCode::ControlEnableMultiplier, nullptr);
	}

	TSound* sound;
	if (someFlag)
	{
		if (someFlag == 1)
		{
			sound = soundwave21;
		}
		else
		{
			if (someFlag < (!pb::FullTiltMode ? 1 : 2) || someFlag > 3)
			{
				auto duration = soundwave41->Play(lite24, "HyperspaceKickOutControl1");
				soundwave36_1->Play(lite24, "HyperspaceKickOutControl2");
				soundwave50_2->Play(lite24, "HyperspaceKickOutControl3");
				lite25->Message(MessageCode::TLightFlasherStartTimed, duration + 5.0f);
				caller->Message(MessageCode::TKickoutRestartTimer, duration);
				return;
			}
			sound = soundwave40;
		}
	}
	else
	{
		switch (activeCount)
		{
		case 1:
			sound = soundwave36_2;
			break;
		case 2:
			sound = soundwave35_2;
			break;
		case 3:
			sound = soundwave38;
			break;
		case 4:
			sound = soundwave39;
			break;
		default:
			sound = soundwave35_1;
			break;
		}
	}
	auto duration = sound->Play(lite24, "HyperspaceKickOutControl4");
	lite25->Message(MessageCode::TLightFlasherStartTimed, 5.0);
	caller->Message(MessageCode::TKickoutRestartTimer, duration);
}

void control::PlungerControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::PlungerFeedBall)
	{
		MissionControl(MessageCode::ControlMissionStarted, nullptr);
		if (easyMode && !block1->ActiveFlag)
			DrainBallBlockerControl(MessageCode::TBlockerEnable, block1);
	}
	else if (code == MessageCode::PlungerStartFeedTimer)
	{
		table_unlimited_balls = false;
		if (!middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0))
			middle_circle->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.0);
		if (!lite200->light_on())
		{
			skill_shot_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite67->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			skill_shot_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.25f);
			l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			l_trek_lights->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.2f);
			l_trek_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.2f);
			r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			r_trek_lights->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.2f);
			r_trek_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.2f);
			TableG->ReflexShotScore = 25000;
			MultiplierLightGroupControl(MessageCode::ControlDisableMultiplier, top_target_lights);
			fuel_bargraph->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite200->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			gate1->Message(MessageCode::TGateDisable, 0.0);
			gate2->Message(MessageCode::TGateDisable, 0.0);
		}
		lite200->MessageField = 0;
	}
}

void control::MedalTargetControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision && !caller->MessageField)
	{
		caller->MessageField = 1;
		if (target6->MessageField
			+ target5->MessageField
			+ target4->MessageField == 3)
		{
			MedalLightGroupControl(MessageCode::TLightGroupResetAndTurnOn, bumber_target_lights);
			int activeCount = bumber_target_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0) - 1;
			const char* text;
			switch (activeCount)
			{
			case 0:
				TableG->AddScore(caller->get_scoring(1));
				text = pb::get_rc_string(Msg::STRING154);
				break;
			case 1:
				TableG->AddScore(caller->get_scoring(2));
				text = pb::get_rc_string(Msg::STRING155);
				break;
			default:
				table_add_extra_ball(4.0);
				text = pb::get_rc_string(Msg::STRING156);
				break;
			}
			info_text_box->Display(text, 2.0);
			target6->MessageField = 0;
			target6->Message(MessageCode::TPopupTargetEnable, 0.0);
			target5->MessageField = 0;
			target5->Message(MessageCode::TPopupTargetEnable, 0.0);
			target4->MessageField = 0;
			target4->Message(MessageCode::TPopupTargetEnable, 0.0);
			return;
		}
		TableG->AddScore(caller->get_scoring(0));
	}
}

void control::MultiplierTargetControl(MessageCode code, TPinballComponent* caller)
{
	if (code == MessageCode::ControlCollision && !caller->MessageField)
	{
		caller->MessageField = 1;
		if (target9->MessageField + target8->MessageField + target7->MessageField == 3)
		{
			TableG->AddScore(caller->get_scoring(1));
			MultiplierLightGroupControl(MessageCode::TLightGroupResetAndTurnOn, top_target_lights);
			int activeCount = top_target_lights->Message(MessageCode::TLightGroupGetOnCount, 0.0);
			const char* text;
			switch (activeCount)
			{
			case 1:
				TableG->ScoreMultiplier = 1;
				text = pb::get_rc_string(Msg::STRING157);
				break;
			case 2:
				TableG->ScoreMultiplier = 2;
				text = pb::get_rc_string(Msg::STRING158);
				break;
			case 3:
				TableG->ScoreMultiplier = 3;
				text = pb::get_rc_string(Msg::STRING159);
				break;
			default:
				TableG->ScoreMultiplier = 4;
				text = pb::get_rc_string(Msg::STRING160);
				break;
			}

			info_text_box->Display(text, 2.0);
			target9->MessageField = 0;
			target9->Message(MessageCode::TPopupTargetEnable, 0.0);
			target8->MessageField = 0;
			target8->Message(MessageCode::TPopupTargetEnable, 0.0);
			target7->MessageField = 0;
			target7->Message(MessageCode::TPopupTargetEnable, 0.0);
		}
		else
		{
			TableG->AddScore(caller->get_scoring(0));
		}
	}
}

void control::BallDrainControl(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code == MessageCode::ControlTimerExpired)
	{
		if (lite199->MessageField)
		{
			TableG->Message(MessageCode::GameOver, 0.0);
			if (pb::chk_highscore())
			{
				soundwave3->Play(nullptr, "BallDrainControl1");
				TableG->LightGroup->Message(MessageCode::TLightFlasherStartTimedThenStayOff, 3.0);
				mission_text_box->Display(pb::get_rc_string(Msg::STRING277), -1.0);
			}
		}
		else
		{
			plunger->Message(MessageCode::PlungerStartFeedTimer, 0.0);
		}
	}
	else if (code == MessageCode::ControlCollision)
	{
		if (table_unlimited_balls)
		{
			drain->Message(MessageCode::Reset, 0.0);
			sink3->Message(MessageCode::TSinkResetTimer, 0.0);
		}
		else
		{
			if (TableG->TiltLockFlag)
			{
				lite200->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite199->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				midi::play_track(MidiTracks::Track1, false);
			}
			if (lite200->light_on())
			{
				soundwave27->Play(nullptr, "BallDrainControl2");
				lite200->Message(MessageCode::TLightResetAndTurnOn, 0.0);
				info_text_box->Display(pb::get_rc_string(Msg::STRING197), -1.0);
				soundwave59->Play(nullptr, "BallDrainControl3");
			}
			else if (lite199->light_on())
			{
				soundwave27->Play(nullptr, "BallDrainControl4");
				lite199->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite200->Message(MessageCode::TLightResetAndTurnOn, 0.0);
				info_text_box->Display(pb::get_rc_string(Msg::STRING196), 2.0);
				soundwave59->Play(nullptr, "BallDrainControl5");
				--TableG->UnknownP78;
			}
			else if (TableG->MultiballCount)
			{
				if (TableG->MultiballCount == 1)
				{
					lite38->Message(MessageCode::TLightResetAndTurnOff, 0.0f);
					lite39->Message(MessageCode::TLightResetAndTurnOff, 0.0f);
					midi::play_track(MidiTracks::Track1, false);
				}
				else if (TableG->MultiballCount == 2)
				{
					lite40->Message(MessageCode::TLightResetAndTurnOff, 0.0f);
				}
			}
			else
			{
				if (!TableG->TiltLockFlag)
				{
					int time = SpecialAddScore(TableG->BonusScore);
					snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING195), time);
					info_text_box->Display(Buffer, 2.0);
				}
				if (TableG->ExtraBalls)
				{
					TableG->ExtraBalls--;

					const char* shootAgainText;
					soundwave59->Play(nullptr, "BallDrainControl7");
					switch (TableG->CurrentPlayer)
					{
					case 0:
						shootAgainText = pb::get_rc_string(Msg::STRING198);
						break;
					case 1:
						shootAgainText = pb::get_rc_string(Msg::STRING199);
						break;
					case 2:
						shootAgainText = pb::get_rc_string(Msg::STRING200);
						break;
					default:
					case 3:
						shootAgainText = pb::get_rc_string(Msg::STRING201);
						break;
					}
					info_text_box->Display(shootAgainText, -1.0);
				}
				else
				{
					TableG->ChangeBallCount(TableG->BallCount - 1);
					if (TableG->CurrentPlayer + 1 != TableG->PlayerCount || TableG->BallCount)
					{
						TableG->Message(MessageCode::SwitchToNextPlayer, 0.0);
						lite199->MessageField = 0;
					}
					else
					{
						lite199->MessageField = 1;
					}
					soundwave27->Play(nullptr, "BallDrainControl8");
				}
				bmpr_inc_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				ramp_bmpr_inc_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite30->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite29->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite1->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite54->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite55->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite56->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite17->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite18->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite27->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite28->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite16->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite20->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				hyper_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite25->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite26->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite130->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite19->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				worm_hole_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				bsink_arrow_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite60->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite59->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite61->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				bumber_target_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				top_target_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				top_circle_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				ramp_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lchute_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				bpr_solotgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite110->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				skill_shot_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite77->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite198->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite196->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite195->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				fuel_bargraph->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				fuel_bargraph->Message(MessageCode::Reset, 0.0);
				GravityWellKickoutControl(MessageCode::Reset, nullptr);
				lite62->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite4->MessageField = 0;
				lite101->MessageField = 0;
				lite102->MessageField = 0;
				lite103->MessageField = 0;
				ramp_tgt_lights->MessageField = 0;
				outer_circle->Message(MessageCode::TLightGroupReset, 0.0);
				middle_circle->Message(MessageCode::TLightGroupReset, 0.0);
				attack_bump->Message(MessageCode::Reset, 0.0);
				launch_bump->Message(MessageCode::Reset, 0.0);
				gate1->Message(MessageCode::Reset, 0.0);
				gate2->Message(MessageCode::Reset, 0.0);
				block1->Message(MessageCode::Reset, 0.0);
				target1->Message(MessageCode::Reset, 0.0);
				target2->Message(MessageCode::Reset, 0.0);
				target3->Message(MessageCode::Reset, 0.0);
				target6->Message(MessageCode::Reset, 0.0);
				target5->Message(MessageCode::Reset, 0.0);
				target4->Message(MessageCode::Reset, 0.0);
				target9->Message(MessageCode::Reset, 0.0);
				target8->Message(MessageCode::Reset, 0.0);
				target7->Message(MessageCode::Reset, 0.0);
				if (lite199->MessageField)
					lite198->MessageField = 32;
				else
					lite198->MessageField = 0;
				MissionControl(MessageCode::ControlMissionComplete, nullptr);
				TableG->Message(MessageCode::ClearTiltLock, 0.0);
				if (lite58->light_on())
					lite58->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				else
					// TableG->BonusScore = 25000;
					TableG->BonusScore = 0;
			}
		}
	}
}


void control::table_control_handler(MessageCode code)
{
	if (code == MessageCode::SetTiltLock)
	{
		table_unlimited_balls = false;
		lite77->Message(MessageCode::TLightFlasherStartTimed, 0.0);
	}
}


void control::AlienMenaceController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::TBumperSetBmpIndex)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			attack_bump->Message(MessageCode::TBumperSetBmpIndex, 0.0);
			l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			l_trek_lights->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.2f);
			l_trek_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.2f);
			r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			r_trek_lights->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.2f);
			r_trek_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.2f);
			lite307->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING275), -1.0);
		return;
	}
	if (bump1 == caller)
	{
		if (bump1->BmpIndex)
		{
			lite307->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 20;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
	}
}

void control::AlienMenacePartTwoController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 8;
			l_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
			l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			r_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
			r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite308->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite311->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING208),
		          lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (bump1 == caller
		|| bump2 == caller
		|| bump3 == caller
		|| bump4 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite308->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite311->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING231), 4.0);
			int addedScore = SpecialAddScore(750000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(7))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "AlienMenacePartTwoController");
			}
		}
	}
}

void control::BlackHoleThreatController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code == MessageCode::TBumperSetBmpIndex)
	{
		if (bump5 == caller)
			MissionControl(MessageCode::ControlMissionStarted, caller);
	}
	else if (code == MessageCode::ControlCollision)
	{
		if (kickout3 == caller
			&& bump5->BmpIndex)
		{
			if (lite316->light_on())
				lite316->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (lite314->light_on())
				lite314->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING225), 4.0);
			int addedScore = SpecialAddScore(1000000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(8))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "BlackHoleThreatController");
			}
		}
	}
	else
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			launch_bump->Message(MessageCode::TBumperSetBmpIndex, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		if (bump5->BmpIndex)
		{
			mission_text_box->Display(pb::get_rc_string(Msg::STRING224), -1.0);
			if (lite316->light_on())
				lite316->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (!lite314->light_on())
			{
				lite314->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			}
		}
		else
		{
			mission_text_box->Display(pb::get_rc_string(Msg::STRING223), -1.0);
			if (lite314->light_on())
				lite314->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (!lite316->light_on())
			{
				lite316->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			}
		}
	}
}

void control::BugHuntController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 15;
			target1->MessageField = 0;
			target1->Message(MessageCode::TPopupTargetEnable, 0.0);
			target2->MessageField = 0;
			target2->Message(MessageCode::TPopupTargetEnable, 0.0);
			target3->MessageField = 0;
			target3->Message(MessageCode::TPopupTargetEnable, 0.0);
			target6->MessageField = 0;
			target6->Message(MessageCode::TPopupTargetEnable, 0.0);
			target5->MessageField = 0;
			target5->Message(MessageCode::TPopupTargetEnable, 0.0);
			target4->MessageField = 0;
			target4->Message(MessageCode::TPopupTargetEnable, 0.0);
			target9->MessageField = 0;
			target9->Message(MessageCode::TPopupTargetEnable, 0.0);
			target8->MessageField = 0;
			target8->Message(MessageCode::TPopupTargetEnable, 0.0);
			target7->MessageField = 0;
			target7->Message(MessageCode::TPopupTargetEnable, 0.0);
			top_circle_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			ramp_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lchute_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			bpr_solotgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite306->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite308->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite310->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite313->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite319->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING226), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (target1 == caller
		|| target2 == caller
		|| target3 == caller
		|| target6 == caller
		|| target5 == caller
		|| target4 == caller
		|| target9 == caller
		|| target8 == caller
		|| target7 == caller
		|| target10 == caller
		|| target11 == caller
		|| target12 == caller
		|| target13 == caller
		|| target14 == caller
		|| target15 == caller
		|| target16 == caller
		|| target17 == caller
		|| target18 == caller
		|| target19 == caller
		|| target20 == caller
		|| target21 == caller
		|| target22 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite306->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite308->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite310->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite313->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite319->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING227), 4.0);
			int addedScore = SpecialAddScore(750000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(7))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "BugHuntController");
			}
		}
	}
}

void control::CosmicPlagueController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 75;
			lite20->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite19->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite305->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite312->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING240), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (flag1 == caller || flag2 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite305->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite312->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 21;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			lite20->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite19->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		}
	}
}

void control::CosmicPlaguePartTwoController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite310->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING241), -1.0);
		return;
	}
	if (roll9 == caller)
	{
		lite310->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite198->MessageField = 1;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
		mission_text_box->Display(pb::get_rc_string(Msg::STRING242), 4.0);
		int addedScore = SpecialAddScore(1750000, true);
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
		if (!AddRankProgress(11))
		{
			mission_text_box->Display(Buffer, 8.0);
			soundwave9->Play(nullptr, "CosmicPlaguePartTwoController");
		}
	}
}

void control::DoomsdayMachineController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 3;
			lite301->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite320->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING238), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (roll4 == caller || roll8 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite301->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite320->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING239), 4.0);
			int addedScore = SpecialAddScore(1250000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(9))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "DoomsdayMachineController");
			}
		}
	}
}

void control::GameoverController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[128];

	if (code == MessageCode::ControlMissionComplete)
	{
		goal_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		pb::mode_change(GameModes::GameOver);
		flip1->Message(MessageCode::GameOver, 0.0);
		flip2->Message(MessageCode::GameOver, 0.0);
		mission_text_box->MessageField = 0;
		midi::play_track(MidiTracks::Track1, false);
		return;
	}
	if (code != MessageCode::ControlMissionStarted)
		return;

	int missionMsg = mission_text_box->MessageField;
	if (missionMsg & 0x100)
	{
		int playerId = missionMsg % 4;
		int playerScore = TableG->PlayerScores[playerId].ScoreStruct->Score;
		auto nextPlayerId = playerId + 1;
		if (playerScore >= 0)
		{
			const char* playerNScoreText = nullptr;
			switch (nextPlayerId)
			{
			case 1:
				playerNScoreText = pb::get_rc_string(Msg::STRING280);
				break;
			case 2:
				playerNScoreText = pb::get_rc_string(Msg::STRING281);
				break;
			case 3:
				playerNScoreText = pb::get_rc_string(Msg::STRING282);
				break;
			case 4:
				playerNScoreText = pb::get_rc_string(Msg::STRING283);
				break;
			default:
				break;
			}
			if (playerNScoreText != nullptr)
			{
				snprintf(Buffer, sizeof Buffer, playerNScoreText, playerScore);
				mission_text_box->Display(Buffer, 3.0);
				int msgField = nextPlayerId == TableG->PlayerCount ? 0x200 : nextPlayerId | 0x100;
				mission_text_box->MessageField = msgField;
				return;
			}
		}
		mission_text_box->MessageField = 0x200;
	}

	if (missionMsg & 0x200)
	{
		int highscoreId = missionMsg % 5;
		int highScore = high_score::highscore_table[highscoreId].Score;
		auto nextHidhscoreId = highscoreId + 1;
		if (highScore > 0)
		{
			const char* highScoreNText = nullptr;
			switch (nextHidhscoreId)
			{
			case 1:
				highScoreNText = pb::get_rc_string(Msg::STRING284);
				break;
			case 2:
				highScoreNText = pb::get_rc_string(Msg::STRING285);
				break;
			case 3:
				highScoreNText = pb::get_rc_string(Msg::STRING286);
				break;
			case 4:
				highScoreNText = pb::get_rc_string(Msg::STRING287);
				break;
			case 5:
				highScoreNText = pb::get_rc_string(Msg::STRING288);
				break;
			default:
				break;
			}
			if (highScoreNText != nullptr)
			{
				snprintf(Buffer, sizeof Buffer, highScoreNText, highScore);
				mission_text_box->Display(Buffer, 3.0);
				int msgField = nextHidhscoreId == 5 ? 0 : nextHidhscoreId | 0x200;
				mission_text_box->MessageField = msgField;
				return;
			}
		}
	}

	mission_text_box->MessageField = 0x100;
	mission_text_box->Display(pb::get_rc_string(Msg::STRING272), 10.0);
}

void control::LaunchTrainingController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite317->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite56->MessageField = 3;
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING211), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (ramp == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite317->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING212), 4.0);
			int addedScore = SpecialAddScore(500000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(6))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "LaunchTrainingController");
			}
		}
	}
}

void control::MaelstromController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 3;
			lite303->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite309->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite315->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING249), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (target1 == caller
		|| target2 == caller
		|| target3 == caller
		|| target6 == caller
		|| target5 == caller
		|| target4 == caller
		|| target9 == caller
		|| target8 == caller
		|| target7 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite303->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite309->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite315->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 25;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
	}
}

void control::MaelstromPartEightController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite130->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite304->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING256), -1.0);
		return;
	}
	if (kickout2 == caller)
	{
		lite304->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite130->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite198->MessageField = 1;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
		int addedScore = SpecialAddScore(5000000, true);
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
		info_text_box->Display(pb::get_rc_string(Msg::STRING149), 4.0);
		if (!AddRankProgress(18))
		{
			mission_text_box->Display(Buffer, 8.0);
			soundwave9->Play(nullptr, "MaelstromPartEightController");
		}
	}
}

void control::MaelstromPartFiveController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite317->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING253), -1.0);
		return;
	}
	if (ramp == caller)
	{
		lite317->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite198->MessageField = 29;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
	}
}

void control::MaelstromPartFourController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 0;
			lite318->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING252), -1.0);
		return;
	}
	if (roll184 == caller)
	{
		lite318->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite198->MessageField = 28;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
	}
}

void control::MaelstromPartSevenController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			AdvanceWormHoleDestination(1);
			sink1->Message(MessageCode::TSinkUnknown7, 0.0);
			sink2->Message(MessageCode::TSinkUnknown7, 0.0);
			sink3->Message(MessageCode::TSinkUnknown7, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING255), -1.0);
		return;
	}
	if (sink1 == caller
		|| sink2 == caller
		|| sink3 == caller)
	{
		lite198->MessageField = 31;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
	}
}

void control::MaelstromPartSixController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite20->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite19->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite305->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite312->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING254), -1.0);
		return;
	}
	if (flag1 == caller || flag2 == caller)
	{
		lite305->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite312->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite198->MessageField = 30;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
		lite20->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite19->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	}
}

void control::MaelstromPartThreeController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 5;
			lite301->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite302->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite307->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite316->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite320->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite321->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING251), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (roll3 == caller
		|| roll2 == caller
		|| roll1 == caller
		|| roll112 == caller
		|| roll111 == caller
		|| roll110 == caller
		|| roll4 == caller
		|| roll8 == caller
		|| roll6 == caller
		|| roll7 == caller
		|| roll5 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite301->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite302->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite307->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite316->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite320->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite321->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 27;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
	}
}

void control::MaelstromPartTwoController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 3;
			lite306->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite308->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite310->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite313->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite319->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING250), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (target10 == caller
		|| target11 == caller
		|| target12 == caller
		|| target13 == caller
		|| target14 == caller
		|| target15 == caller
		|| target16 == caller
		|| target17 == caller
		|| target18 == caller
		|| target19 == caller
		|| target20 == caller
		|| target21 == caller
		|| target22 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite306->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite308->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite310->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite313->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite319->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 26;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
	}
}

void control::PracticeMissionController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite308->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite311->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite56->MessageField = 8;
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING208), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}

	if (bump1 == caller
		|| bump2 == caller
		|| bump3 == caller
		|| bump4 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite308->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite311->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING209), 4.0);
			int addedScore = SpecialAddScore(500000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(6))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "PracticeMissionController");
			}
		}
	}
}

void control::ReconnaissanceController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 15;
			lite301->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite302->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite307->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite316->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite320->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite321->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING235), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (roll3 == caller
		|| roll2 == caller
		|| roll1 == caller
		|| roll112 == caller
		|| roll111 == caller
		|| roll110 == caller
		|| roll4 == caller
		|| roll8 == caller
		|| roll6 == caller
		|| roll7 == caller
		|| roll5 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, nullptr);
		}
		else
		{
			lite301->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite302->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite307->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite316->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite320->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite321->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING237), 4.0);
			int addedScore = SpecialAddScore(1250000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(9))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "ReconnaissanceController");
			}
		}
	}
}

void control::ReentryTrainingController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 3;
			l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			l_trek_lights->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.2f);
			l_trek_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.2f);
			r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			r_trek_lights->Message(MessageCode::TLightGroupOffsetAnimationForward, 0.2f);
			r_trek_lights->Message(MessageCode::TLightGroupAnimationBackward, 0.2f);
			lite307->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING213), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (roll3 == caller
		|| roll2 == caller
		|| roll1 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite307->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING214), 4.0);
			int addedScore = SpecialAddScore(500000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(6))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "ReentryTrainingController");
			}
		}
	}
}

void control::RescueMissionController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	switch (code)
	{
	case MessageCode::ControlCollision:
		{
			if (target1 == caller
				|| target2 == caller
				|| target3 == caller)
			{
				MissionControl(MessageCode::ControlMissionStarted, caller);
				return;
			}
			if (kickout2 != caller || !lite20->light_on())
				return;
			lite56->MessageField = lite56->MessageField - 1;
			if (lite56->MessageField)
			{
				MissionControl(MessageCode::ControlMissionStarted, caller);
				return;
			}
			if (lite303->light_on())
				lite303->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (lite304->light_on())
				lite304->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING230), 4.0);
			int addedScore = SpecialAddScore(750000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(7))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "RescueMissionController");
			}
			break;
		}
	case MessageCode::ControlMissionComplete:
		lite20->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite19->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite56->MessageField = 1;
		break;
	case MessageCode::ControlMissionStarted:
		if (lite20->light_on())
		{
			mission_text_box->Display(pb::get_rc_string(Msg::STRING229), -1.0);
			if (lite303->light_on())
				lite303->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (!lite304->light_on())
			{
				lite304->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			}
		}
		else
		{
			mission_text_box->Display(pb::get_rc_string(Msg::STRING228), -1.0);
			if (lite304->light_on())
				lite304->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (!lite303->light_on())
			{
				lite303->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			}
		}
		break;
	default:
		break;
	}
}

void control::SatelliteController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 3;
			lite308->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING233), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (bump4 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite308->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING234), 4.0);
			int addedScore = SpecialAddScore(1250000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(9))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "SatelliteController");
			}
		}
	}
}

void control::ScienceMissionController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 9;
			target1->MessageField = 0;
			target1->Message(MessageCode::TPopupTargetEnable, 0.0);
			target2->MessageField = 0;
			target2->Message(MessageCode::TPopupTargetEnable, 0.0);
			target3->MessageField = 0;
			target3->Message(MessageCode::TPopupTargetEnable, 0.0);
			target6->MessageField = 0;
			target6->Message(MessageCode::TPopupTargetEnable, 0.0);
			target5->MessageField = 0;
			target5->Message(MessageCode::TPopupTargetEnable, 0.0);
			target4->MessageField = 0;
			target4->Message(MessageCode::TPopupTargetEnable, 0.0);
			target9->MessageField = 0;
			target9->Message(MessageCode::TPopupTargetEnable, 0.0);
			target8->MessageField = 0;
			target8->Message(MessageCode::TPopupTargetEnable, 0.0);
			target7->MessageField = 0;
			target7->Message(MessageCode::TPopupTargetEnable, 0.0);
			lite303->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite309->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite315->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING215), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (target1 == caller
		|| target2 == caller
		|| target3 == caller
		|| target6 == caller
		|| target5 == caller
		|| target4 == caller
		|| target9 == caller
		|| target8 == caller
		|| target7 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite303->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite309->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite315->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING216), 4.0);
			int addedScore = SpecialAddScore(750000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(9))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "ScienceMissionController");
			}
		}
	}
}

void control::SecretMissionGreenController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite6->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite2->Message(MessageCode::TLightSetOnStateBmpIndex, 1.0);
			lite2->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite2->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		const char* v2 = pb::get_rc_string(Msg::STRING245);
		mission_text_box->Display(v2, -1.0);
		return;
	}
	if (sink2 == caller)
	{
		lite198->MessageField = 1;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
		mission_text_box->Display(pb::get_rc_string(Msg::STRING246), 4.0);
		int addedScore = SpecialAddScore(1500000, true);
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
		if (!AddRankProgress(10))
		{
			mission_text_box->Display(Buffer, 8.0);
			soundwave9->Play(nullptr, "SecretMissionGreenController");
		}
	}
}

void control::SecretMissionRedController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite5->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite4->Message(MessageCode::TLightSetOnStateBmpIndex, 2.0);
			lite4->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite4->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING244), -1.0);
		return;
	}
	if (sink1 == caller)
	{
		lite198->MessageField = 23;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
	}
}

void control::SecretMissionYellowController(MessageCode code, TPinballComponent* caller)
{
	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			worm_hole_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			bsink_arrow_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			bsink_arrow_lights->Message(MessageCode::TLightSetMessageField, 0.0);
			lite110->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite7->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite3->Message(MessageCode::TLightSetOnStateBmpIndex, 0.0);
			lite3->Message(MessageCode::TLightResetAndTurnOn, 0.0);
			lite3->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING243), -1.0);
		return;
	}
	if (sink3 == caller)
	{
		lite198->MessageField = 22;
		MissionControl(MessageCode::ControlMissionComplete, nullptr);
	}
}

void control::SelectMissionController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	switch (code)
	{
	case MessageCode::TLightGroupToggleSplitIndex:
	case MessageCode::TLightGroupCountdownEnded:
		if (fuel_bargraph != caller)
			return;
		MissionControl(MessageCode::ControlMissionStarted, caller);
		return;
	case MessageCode::ControlCollision:
		{
			int missionLevel = 0;
			if (target13 == caller)
				missionLevel = 1;
			if (target14 == caller)
				missionLevel = 2;
			if (target15 == caller)
				missionLevel = 3;
			if (!missionLevel)
			{
				if (ramp == caller
					&& lite56->light_on()
					&& fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0))
				{
					lite56->Message(MessageCode::TLightResetAndTurnOff, 0.0);
					lite198->Message(MessageCode::TLightResetAndTurnOn, 0.0);
					outer_circle->Message(MessageCode::TLightGroupAnimationBackward, -1.0);
					if (lite317->light_on())
						lite317->Message(MessageCode::TLightResetAndTurnOff, 0.0);
					if (lite318->light_on())
						lite318->Message(MessageCode::TLightResetAndTurnOff, 0.0);
					if (lite319->light_on())
						lite319->Message(MessageCode::TLightResetAndTurnOff, 0.0);
					lite198->MessageField = lite56->MessageField;
					auto scoreId = lite56->MessageField - 2;
					MissionControl(MessageCode::ControlMissionComplete, nullptr);
					auto score = !pb::FullTiltMode ? mission_select_scores[scoreId] : 100000;
					int addedScore = SpecialAddScore(score);
					snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING178), addedScore);
					mission_text_box->Display(Buffer, 4.0);
					midi::play_track(MidiTracks::Track2, true);
				}
				return;
			}

			if (lite101->MessageField == 7)
			{
				lite101->MessageField = 0;
				missionLevel = 4;
			}

			int missionId;
			auto activeCount = middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0);
			switch (activeCount)
			{
			case 1:
				switch (missionLevel)
				{
				case 1:
					missionId = 3;
					break;
				case 2:
					missionId = 4;
					break;
				case 3:
					missionId = 2;
					break;
				default:
					missionId = 5;
					break;
				}
				break;
			case 2:
			case 3:
				switch (missionLevel)
				{
				case 1:
					missionId = 9;
					break;
				case 2:
					missionId = 11;
					break;
				case 3:
					missionId = 10;
					break;
				default:
					missionId = 16;
					break;
				}
				break;
			case 4:
			case 5:
				switch (missionLevel)
				{
				case 1:
					missionId = 6;
					break;
				case 2:
					missionId = 8;
					break;
				case 3:
					missionId = 7;
					break;
				default:
					missionId = 15;
					break;
				}
				break;
			case 6:
			case 7:
				switch (missionLevel)
				{
				case 1:
					missionId = 12;
					break;
				case 2:
					missionId = 13;
					break;
				case 3:
					missionId = 14;
					break;
				default:
					missionId = 17;
					break;
				}
				break;
			case 8:
			case 9:
				switch (missionLevel)
				{
				case 1:
					missionId = 15;
					break;
				case 2:
					missionId = 16;
					break;
				case 3:
					missionId = 17;
					break;
				default:
					missionId = 18;
					break;
				}
				break;
			default:
				return;
			}
			lite56->MessageField = missionId;
			lite56->Message(MessageCode::TLightFlasherStartTimedThenStayOn, 2.0);
			lite198->Message(MessageCode::TLightFlasherStart, 0.0);
			MissionControl(MessageCode::ControlMissionStarted, caller);
			return;
		}
	case MessageCode::ControlMissionComplete:
		midi::play_track(MidiTracks::Track1, false);
		lite198->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		outer_circle->Message(MessageCode::TLightGroupReset, 0.0);
		ramp_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		lite56->MessageField = 0;
		lite101->MessageField = 0;
		l_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
		l_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		r_trek_lights->Message(MessageCode::TLightGroupReset, 0.0);
		r_trek_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		goal_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		worm_hole_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		bsink_arrow_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		break;
	case MessageCode::ControlMissionStarted:
		break;
	default:
		return;
	}

	if (fuel_bargraph->Message(MessageCode::TLightGroupGetOnCount, 0.0))
	{
		if (lite56->light_on() && lite56->MessageField >= 2)
		{
			auto missionText = pb::get_rc_string(MissionRcArray[lite56->MessageField - 2]);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING207), missionText);
			mission_text_box->Display(Buffer, -1.0);
			if (lite318->light_on())
				lite318->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (lite319->light_on())
				lite319->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (!lite317->light_on())
				lite317->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else
		{
			mission_text_box->Display(pb::get_rc_string(Msg::STRING205), -1.0);
			if (lite317->light_on())
				lite317->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (lite318->light_on())
				lite318->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			if (!lite319->light_on())
			{
				lite319->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			}
		}
	}
	else
	{
		mission_text_box->Display(pb::get_rc_string(Msg::STRING206), -1.0);
		if (lite317->light_on())
			lite317->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		if (lite319->light_on())
			lite319->Message(MessageCode::TLightResetAndTurnOff, 0.0);
		if (!lite318->light_on())
		{
			lite318->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
	}
}

void control::SpaceRadiationController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code == MessageCode::ControlCollision)
	{
		if (target16 == caller
			|| target17 == caller
			|| target18 == caller)
		{
			if (lite104->MessageField == 7)
			{
				lite104->MessageField = 15;
				bsink_arrow_lights->Message(MessageCode::TLightFlasherStartTimed, 0.0);
				lite313->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				MissionControl(MessageCode::ControlMissionStarted, caller);
				AdvanceWormHoleDestination(1);
			}
		}
		else if ((sink1 == caller
				|| sink2 == caller
				|| sink3 == caller)
			&& lite104->MessageField == 15)
		{
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING222), 4.0);
			int addedScore = SpecialAddScore(1000000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(8))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "SpaceRadiationController");
			}
		}
	}
	else
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lchute_tgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite104->MessageField = 0;
			lite313->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code == MessageCode::ControlMissionStarted)
		{
			const char* text;
			if (lite104->MessageField == 15)
				text = pb::get_rc_string(Msg::STRING221);
			else
				text = pb::get_rc_string(Msg::STRING276);
			mission_text_box->Display(text, -1.0);
		}
	}
}

void control::StrayCometController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code == MessageCode::ControlCollision)
	{
		if (target19 == caller
			|| target20 == caller
			|| target21 == caller)
		{
			if (lite107->MessageField == 7)
			{
				lite306->Message(MessageCode::TLightResetAndTurnOff, 0.0);
				lite304->Message(MessageCode::TLightFlasherStartTimed, 0.0);
				lite107->MessageField = 15;
				MissionControl(MessageCode::ControlMissionStarted, caller);
			}
		}
		else if (kickout2 == caller && lite107->MessageField == 15)
		{
			lite304->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
			mission_text_box->Display(pb::get_rc_string(Msg::STRING220), 4.0);
			int addedScore = SpecialAddScore(1000000, true);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING179), addedScore);
			if (!AddRankProgress(8))
			{
				mission_text_box->Display(Buffer, 8.0);
				soundwave9->Play(nullptr, "StrayCometController");
			}
		}
	}
	else
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			bpr_solotgt_lights->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite107->MessageField = 0;
			lite306->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code == MessageCode::ControlMissionStarted)
		{
			const char* text;
			if (lite107->MessageField == 15)
				text = pb::get_rc_string(Msg::STRING219);
			else
				text = pb::get_rc_string(Msg::STRING218);
			mission_text_box->Display(text, -1.0);
		}
	}
}

void control::TimeWarpController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite56->MessageField = 25;
			lite300->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite322->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING247), lite56->MessageField);
		mission_text_box->Display(Buffer, -1.0);
		return;
	}
	if (rebo1 == caller
		|| rebo2 == caller
		|| rebo3 == caller
		|| rebo4 == caller)
	{
		lite56->MessageField = lite56->MessageField - 1;
		if (lite56->MessageField)
		{
			MissionControl(MessageCode::ControlMissionStarted, caller);
		}
		else
		{
			lite300->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite322->Message(MessageCode::TLightResetAndTurnOff, 0.0);
			lite198->MessageField = 24;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
	}
}

void control::TimeWarpPartTwoController(MessageCode code, TPinballComponent* caller)
{
	char Buffer[64];

	if (code != MessageCode::ControlCollision)
	{
		if (code == MessageCode::ControlMissionComplete)
		{
			lite55->Message(MessageCode::TLightFlasherStartTimed, -1.0);
			lite26->Message(MessageCode::TLightFlasherStartTimed, -1.0);
			lite304->Message(MessageCode::TLightFlasherStartTimed, 0.0);
			lite317->Message(MessageCode::TLightFlasherStartTimed, 0.0);
		}
		else if (code != MessageCode::ControlMissionStarted)
		{
			return;
		}
		mission_text_box->Display(pb::get_rc_string(Msg::STRING248), -1.0);
		return;
	}
	if (kickout2 == caller)
	{
		mission_text_box->Display(pb::get_rc_string(Msg::STRING148), 4.0);
		if (middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0) > 1)
		{
			middle_circle->Message(MessageCode::TLightGroupOffsetAnimationBackward, 5.0);
			int rank = middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING274), pb::get_rc_string(RankRcArray[rank - 1]));
			mission_text_box->Display(Buffer, 8.0);
		}
	}
	else
	{
		if (ramp != caller)
			return;
		mission_text_box->Display(pb::get_rc_string(Msg::STRING147), 4.0);
		if (middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0) < 9)
		{
			int rank = middle_circle->Message(MessageCode::TLightGroupGetOnCount, 0.0);
			middle_circle->Message(MessageCode::TLightGroupResetAndTurnOn, 5.0);
			snprintf(Buffer, sizeof Buffer, pb::get_rc_string(Msg::STRING273), pb::get_rc_string(RankRcArray[rank]));
		}
		if (!AddRankProgress(12))
		{
			mission_text_box->Display(Buffer, 8.0);
			soundwave10->Play(nullptr, "TimeWarpPartTwoController");
		}
	}
	SpecialAddScore(2000000);
	lite55->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	lite26->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	lite304->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	lite317->Message(MessageCode::TLightResetAndTurnOff, 0.0);
	lite198->MessageField = 1;
	MissionControl(MessageCode::ControlMissionComplete, nullptr);
	// SpecialAddScore sets the score dirty flag. So next tick it will be redrawn.
}

void control::UnselectMissionController(MessageCode code, TPinballComponent* caller)
{
	lite198->MessageField = 1;
	MissionControl(MessageCode::ControlMissionComplete, nullptr);
}

void control::WaitingDeploymentController(MessageCode code, TPinballComponent* caller)
{
	switch (code)
	{
	case MessageCode::ControlCollision:
		if (oneway4 == caller || oneway10 == caller)
		{
			lite198->MessageField = 1;
			MissionControl(MessageCode::ControlMissionComplete, nullptr);
		}
		break;
	case MessageCode::ControlMissionComplete:
		mission_text_box->Clear();
		waiting_deployment_flag = 0;
		midi::play_track(MidiTracks::Track1, false);
		break;
	case MessageCode::ControlMissionStarted:
		mission_text_box->Display(pb::get_rc_string(Msg::STRING151), -1.0);
		break;
	default:
		break;
	}
}
