PP_PLAYER = [
    "speed", "acceleration", "sa", "distance", "orientation", "direction",
    "x_position", "y_position",
    "speed_lag1", "speed_lag3", "speed_d1", "speed_d3", "speed_d5",
    "acceleration_lag1", "accel_d1", "sa_lag1", "sa_lag3", "sa_d1", "sa_d3",
    "dir_change1", "dir_change3", "ori_change1", "ori_change3",
    "speed_roll5_mean", "speed_roll5_max", "speed_roll5_min",
    "acceleration_roll5_mean", "acceleration_roll5_max",
    "sa_roll5_mean", "sa_roll5_min", "sa_roll5_max",
    "speed_fut3_mean", "speed_fut5_mean", "speed_lead1",
    "team_home", "position_code",
]
PP_PLAYER_HELMET = [
    "h_top", "h_w", "h_h", "h_cnt", "h_cy", "h_area",
    "top_rank", "cy_rank", "top_minus_med",
]
PP_PAIR = [
    "dist", "dx_field", "dy_field", "same_team", "speed_sum", "speed_diff",
    "accel_sum", "accel_diff", "sa_sum", "sa_min", "dir_diff", "ori_diff",
    "dir1_to_line", "dir2_to_line", "ori1_to_line", "closing_speed", "rel_speed",
    "dist_lag1", "dist_lag3", "dist_lead1", "dist_lead3", "dist_d1", "dist_d3",
    "dist_fd1", "dist_fd3", "dist_roll3_min", "dist_roll5_min", "dist_roll5_mean",
    "dist_fut3_min", "dist_fut5_min", "dist_fut5_mean",
]
PP_HELMET_PAIR = [
    "sid_pair_dpx", "sid_pair_dpx_norm", "sid_pair_iou",
    "end_pair_dpx", "end_pair_dpx_norm", "end_pair_iou",
    "pair_dpx_min", "pair_dpx_mean", "pair_iou_max",
    "views_both", "sid_both", "end_both",
    "pdpx_lag1", "pdpx_lead1", "pdpx_win5_min", "pdpx_win5_mean", "pdpx_win9_min",
]

def pp_features():
    f = []
    for c in PP_PLAYER:
        f += [f"{c}_1", f"{c}_2"]
    for c in PP_PLAYER_HELMET:
        f += [f"sid_{c}_1", f"end_{c}_1", f"sid_{c}_2", f"end_{c}_2"]
    f += PP_PAIR + PP_HELMET_PAIR
    f += ["step"]
    return f

GROUND_PLAYER = PP_PLAYER + ["jersey"]
GROUND_HELMET = PP_PLAYER_HELMET + ["h_bottom", "h_left", "area_rank"]

def ground_features():
    f = [f"{c}_1" for c in GROUND_PLAYER]
    for c in GROUND_HELMET:
        f += [f"sid_{c}_1", f"end_{c}_1"]
    f += ["step", "views_both"]
    return f

V2_PAIR = ["log_dist", "inv_dist", "time_to_collision", "close_per_speed",
           "dist_x_closing", "pair_third_min", "pair_c2_1", "pair_c2_2",
           "pair_is_nn1", "pair_is_nn2"]
V2_PLAYER = ["nb1", "nb2", "opp1", "n2", "n5", "nn_mutual",
             "nb1_lag1", "nb1_lead1", "nb1_d1", "nb1_fd1", "n2_lead1"]
V2_HELMET = ["sid_h_top_d1", "sid_h_top_fd1", "sid_h_top_fd3", "sid_h_top_pmed",
             "end_h_top_d1", "end_h_top_fd1", "end_h_top_pmed",
             "sid_h_cy_fd1", "end_h_cy_fd1", "helmet_n_views"]

def pp_features_v2():
    f = pp_features()
    f += V2_PAIR
    for c in V2_PLAYER:
        f += [f"{c}_1", f"{c}_2"]
    for c in V2_HELMET:
        f += [f"{c}_1", f"{c}_2"]
    return f

def ground_features_v2():
    f = ground_features()
    f += ["log_dist", "inv_dist", "pair_third_min"]
    for c in V2_PLAYER:
        f += [f"{c}_1"]
    for c in V2_HELMET:
        f += [f"{c}_1"]
    # extra ground-specific helmet temporal
    f += ["sid_h_h_fd1_1", "end_h_h_fd1_1", "sid_h_top_lag3_1", "end_h_top_lag3_1",
          "sid_h_top_lead3_1", "end_h_top_lead3_1", "nb3_1", "nb1_lead3_1"]
    return f
