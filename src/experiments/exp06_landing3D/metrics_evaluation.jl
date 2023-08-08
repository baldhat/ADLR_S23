using CSV
using DataFrames
using Statistics
using Plots
using Printf
using StatsPlots

# Load the data
df_without_acc = CSV.read("./src/experiments/exp06_landing3D/runs_without_accel/performance_metrics_1000runs.csv",
                          DataFrame)
df_with_acc = CSV.read("./src/experiments/exp06_landing3D/runs_with_accel/landing3D_3600000.csv",
                       DataFrame)
# df_without_acc = df_without_acc[df_without_acc.position_error .< 1, :]
# df_with_acc = df_with_acc[df_with_acc.position_error .< 1, :]

println("Outliers with acceleration: ", sum(df_with_acc.position_error .> 1))
println("Outliers without acceleration: ", sum(df_without_acc.position_error .> 1))

wind_thr = 4
df_with_acc_lowwind = df_with_acc[df_with_acc.wind_mag .< wind_thr, :]
df_without_acc_lowwind = df_without_acc[df_without_acc.wind_mag .< wind_thr, :]
df_with_acc_highwind = df_with_acc[df_with_acc.wind_mag .>= wind_thr, :]
df_without_acc_highwind = df_without_acc[df_without_acc.wind_mag .>= wind_thr, :]
println("Otliers with acceleration (low wind): ", sum(df_with_acc_lowwind.position_error .> 1))
println("Otliers without acceleration (low wind): ", sum(df_without_acc_lowwind.position_error .> 1))
println("Otliers with acceleration (high wind): ", sum(df_with_acc_highwind.position_error .> 1))
println("Otliers without acceleration (high wind): ", sum(df_without_acc_highwind.position_error .> 1))

# Compute the mean and standard deviation
# position error
errpos_wo_mean, errpos_wo_std = mean(df_without_acc.position_error), std(df_without_acc.position_error)
errpos_wo_lowwind_mean, errpos_wo_lowwind_std = mean(df_without_acc_lowwind.position_error), std(df_without_acc_lowwind.position_error)
errpos_wo_highwind_mean, errpos_wo_highwind_std = mean(df_without_acc_highwind.position_error), std(df_without_acc_highwind.position_error)
errpos_w_mean, errpos_w_std = mean(df_with_acc.position_error), std(df_with_acc.position_error)
errpos_w_lowwind_mean, errpos_w_lowwind_std = mean(df_with_acc_lowwind.position_error), std(df_with_acc_lowwind.position_error)
errpos_w_highwind_mean, errpos_w_highwind_std = mean(df_with_acc_highwind.position_error), std(df_with_acc_highwind.position_error)


p1 = bar(["Without\nAccel", "With\nAccel"], [errpos_wo_mean, errpos_w_mean],
        #  yerr = [errpos_wo_std, errpos_w_std],
         fill=["orangered", "green4"], bar_width=0.9, alpha=0.7,
         text=[" ", text(@sprintf("%.1f", (errpos_w_mean/errpos_wo_mean-1)*100)*" %", 10, :bottom)],
         title = "Position Error", ylabel = "Mean Error (m)", legend = false)
          
p5 = bar(["low\nw/o", "low\nw/", "high\nw/o", "high\nw/"], 
         [errpos_wo_lowwind_mean, errpos_w_lowwind_mean, errpos_wo_highwind_mean, errpos_w_highwind_mean],
        #  yerr = [errpos_wo_lowwind_std, errpos_w_lowwind_std, errpos_wo_highwind_std, errpos_w_highwind_std],
         text=["", text(@sprintf("%.1f", (errpos_w_lowwind_mean/errpos_wo_lowwind_mean-1)*100)*" %", 6, :bottom),
               "", text(@sprintf("%.1f", (errpos_w_highwind_mean/errpos_wo_highwind_mean-1)*100)*" %", 6, :bottom)],
         fill=["firebrick3", "coral", "seagreen", "darkseagreen1"], bar_width=0.9, alpha=0.7,
         ylabel = "Mean Error (m)", legend = false)


# rotation error
errori_wo_mean, errori_wo_std = mean(df_without_acc.rotation_error), std(df_without_acc.rotation_error)
errori_wo_lowwind_mean, errori_wo_lowwind_std = mean(df_without_acc_lowwind.rotation_error), std(df_without_acc_lowwind.rotation_error)
errori_wo_highwind_mean, errori_wo_highwind_std = mean(df_without_acc_highwind.rotation_error), std(df_without_acc_highwind.rotation_error)
errori_w_mean, errori_w_std = mean(df_with_acc.rotation_error), std(df_with_acc.rotation_error)
errori_w_lowwind_mean, errori_w_lowwind_std = mean(df_with_acc_lowwind.rotation_error), std(df_with_acc_lowwind.rotation_error)
errori_w_highwind_mean, errori_w_highwind_std = mean(df_with_acc_highwind.rotation_error), std(df_with_acc_highwind.rotation_error)
p2 = bar(["Without\nAccel", "With\nAccel"], [errori_wo_mean, errori_w_mean],
        #  yerr = [errori_wo_std, errori_w_std],
         fill=["orangered", "green4"], bar_width=0.9, alpha=0.7,
         text=["", text(@sprintf("%.1f", (errori_w_mean/errori_wo_mean-1)*100)*" %", 10, :bottom)],
         title = "Rotation Error", ylabel = "Mean Error (°)", legend = false)
p6 = bar(["low\nw/o", "low\nw/", "high\nw/o", "high\nw/"], [errori_wo_lowwind_mean, errori_w_lowwind_mean, errori_wo_highwind_mean, errori_w_highwind_mean],
        #  yerr = [errori_wo_lowwind_std, errori_w_lowwind_std, errori_wo_highwind_std, errori_w_highwind_std],
         text=["", text(@sprintf("%.1f", (errori_w_lowwind_mean/errori_wo_lowwind_mean-1)*100)*" %", 6, :bottom),
               "", text(@sprintf("%.1f", (errori_w_highwind_mean/errori_wo_highwind_mean-1)*100)*" %", 6, :bottom)],
         fill=["firebrick3", "coral", "seagreen", "darkseagreen1"], bar_width=0.9, alpha=0.7,
         ylabel = "Mean Error (°)", legend = false)



# Return
Return_wo_mean, Return_wo_std = mean(df_without_acc.Return), std(df_without_acc.Return)
Return_wo_lowwind_mean, Return_wo_lowwind_std = mean(df_without_acc_lowwind.Return), std(df_without_acc_lowwind.Return)
Return_wo_highwind_mean, Return_wo_highwind_std = mean(df_without_acc_highwind.Return), std(df_without_acc_highwind.Return)
Return_w_mean, Return_w_std = mean(df_with_acc.Return), std(df_with_acc.Return)
Return_w_lowwind_mean, Return_w_lowwind_std = mean(df_with_acc_lowwind.Return), std(df_with_acc_lowwind.Return)
Return_w_highwind_mean, Return_w_highwind_std = mean(df_with_acc_highwind.Return), std(df_with_acc_highwind.Return)
p3 = bar(["Without\nAccel", "With\nAccel"], [Return_wo_mean, Return_w_mean],
        #  yerr = [Return_wo_std, Return_w_std],
         fill=["orangered", "green4"], bar_width=0.9, alpha=0.7,
         text=["", text(@sprintf("+%.1f", (Return_w_mean/Return_wo_mean-1)*100)*" %", 10, :bottom)],
         title = "Return", ylabel = "Mean Return", legend = false)
p7 = bar(["low\nw/o", "low\nw/", "high\nw/o", "high\nw/"], [Return_wo_lowwind_mean, Return_w_lowwind_mean, Return_wo_highwind_mean, Return_w_highwind_mean],
        #  yerr = [Return_wo_lowwind_std, Return_w_lowwind_std, Return_wo_highwind_std, Return_w_highwind_std],        
         text=["", text(@sprintf("+%.1f", (Return_w_lowwind_mean/Return_wo_lowwind_mean-1)*100)*" %", 6, :bottom),
               "", text(@sprintf("+%.1f", (Return_w_highwind_mean/Return_wo_highwind_mean-1)*100)*" %", 6, :bottom)],
         fill=["firebrick3", "coral", "seagreen", "darkseagreen1"], bar_width=0.9, alpha=0.7,
         ylabel = "Mean Return", legend = false)

# Outliers
rot_thr = 35
pos_thr = 1.
# outliers_wo = sum(df_without_acc.rotation_error .> rot_thr) / nrow(df_without_acc) * 100
# outliers_wo_lowwind = sum(df_without_acc_lowwind.rotation_error .> rot_thr) / nrow(df_without_acc_lowwind) * 100
# outliers_wo_highwind = sum(df_without_acc_highwind.rotation_error .> rot_thr) / nrow(df_without_acc_highwind) * 100
# outliers_w = sum(df_with_acc.rotation_error .> rot_thr) / nrow(df_with_acc) * 100
# outliers_w_lowwind = sum(df_with_acc_lowwind.rotation_error .> rot_thr) / nrow(df_with_acc_lowwind) * 100
# outliers_w_highwind = sum(df_with_acc_highwind.rotation_error .> rot_thr) / nrow(df_with_acc_highwind) * 100
outliers_wo = sum(df_without_acc.rotation_error .> rot_thr * df_without_acc.position_error .> pos_thr) / nrow(df_without_acc) * 100
outliers_wo_lowwind = sum(df_without_acc_lowwind.rotation_error .> rot_thr * df_without_acc_lowwind.position_error .> pos_thr) / nrow(df_without_acc_lowwind) * 100
outliers_wo_highwind = sum(df_without_acc_highwind.rotation_error .> rot_thr * df_without_acc_highwind.position_error .> pos_thr) / nrow(df_without_acc_highwind) * 100
outliers_w = sum(df_with_acc.rotation_error .> rot_thr * df_with_acc.position_error .> pos_thr) / nrow(df_with_acc) * 100
outliers_w_lowwind = sum(df_with_acc_lowwind.rotation_error .> rot_thr * df_with_acc_lowwind.position_error .> pos_thr) / nrow(df_with_acc_lowwind) * 100
outliers_w_highwind = sum(df_with_acc_highwind.rotation_error .> rot_thr * df_with_acc_highwind.position_error .> pos_thr) / nrow(df_with_acc_highwind) * 100
p4 = bar(["Without\nAccel", "With\nAccel"], [outliers_wo, outliers_w],
         fill=["orangered", "green4"], bar_width=0.9, alpha=0.7,
         text=["", text(@sprintf("%.1f", (outliers_w/outliers_wo-1)*100)*" %", 10, :bottom)],
         title = "Outliers", ylabel = "Outlier Rate (%)", legend = false)
p8 = bar(["low\nw/o", "low\nw/", "high\nw/o", "high\nw/"], [outliers_wo_lowwind, outliers_w_lowwind, outliers_wo_highwind, outliers_w_highwind],
         text=["", text(@sprintf("%.1f", (outliers_w_lowwind/outliers_wo_lowwind-1)*100)*" %", 6, :bottom),
             "", text(@sprintf("%.1f", (outliers_w_highwind/outliers_wo_highwind-1)*100)*" %", 6, :bottom)],
         fill=["firebrick3", "coral", "seagreen", "darkseagreen1"], bar_width=0.9, alpha=0.7,
         ylabel = "Outlier Rate (%)", legend = false)

plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = (2,4), size = (850, 600), dpi = 300)
