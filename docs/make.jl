# Run this file to build a new documentation
using Documenter
cd("src")
include("../src/VtolModel.jl")
include("../src/RigidBodies.jl")
include("../src/Visualization.jl")
include("../src/VtolOptimization.jl")
include("../src/Px4Utils.jl")
include("../src/VtolParameterOptimisation.jl")
using Main.VtolModel



makedocs(   sitename="Flyonic",
            format = Documenter.HTML(prettyurls = false),
            pages = [
                    "index.md",
                    "Quickstart guide" => "quick_start.md",
                    "Examples" => "examples.md",
                    "Julia Modules" => [  "VTOL Model" => "vtol_model.md",
                                    "ETH Controller" => "global_controller.md",
                                    "Rigid Body" => "rigid_body.md",
                     #               "Optimization" => "VtolOptimization.md",   
                                    "Visualization" => "visualization.md",   
                                    "Px4 Utils" => "px4_utils.md",   
                                    "VTOL Model Parameter Optimisation" => "vtol_parameter_optimisation.md",],
                    "Unittest" => "unittest.md",
                    "Hardware" => [ "overview" => "hardware/hardware.md",
                                    "IRON bird quickstart" => "hardware/iron_bird.md",
                                    "Ubuntu configurations" => "hardware/installation.md",
                                    "PX4 customization" => "hardware/px4_ext_control.md", 
                                    "VTOL3 Brett" => "hardware/vtol3_brett.md",
                                    "Microcomputer" => "hardware/microcomputer.md",],
                     "Flight experiments" => [ "Airfields" => "experiments/airfields.md",
                                               "Flight laws DE" => "experiments/flight_laws.md",],
                     "Related Work" => "related_work.md",
                     "Workflow & Commands" => [# "Telegram Bot" => "workflow/telegram_bot.md",
                                                "Remote execution" => "workflow/executing_remotely.md",
                                                #"terminal commands" => "workflow/terminal_commands.md",
                                                ],
                    ],
        )


# TODO: implement Auto FTP Upload to docu.flyonic.de
# https://github.com/invenia/FTPClient.jl