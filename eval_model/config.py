# config.py
# エネルギーシミュレーターの設定値

params = {
    "capacity_ashp": 100.0,   # ASHP1の定格熱出力 [kW]
    "cop_ashp": 4.0,           # ASHP1のCOP [-]（一定）
    "capacity_tes": 300.0,   # TESの容量 [kWh]
    "unserved_load_penalty": 10.0,  # TESが負荷を供給できない場合のペナルティ係数 [kg-CO2換算]
}
