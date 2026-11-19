CREATE TABLE IF NOT EXISTS test_backend_ops (
  test_time TEXT,
  build_commit TEXT,
  backend_name TEXT,
  op_name TEXT,
  op_params TEXT,
  test_mode TEXT,
  supported INTEGER,
  passed INTEGER,
  error_message TEXT,
  time_us REAL,
  flops REAL,
  bandwidth_gb_s REAL,
  memory_kb INTEGER,
  n_runs INTEGER,
  device_description TEXT,
  backend_reg_name TEXT
);

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:27Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,1,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '0.833511', '0.000000', '54.924425', '48', '1204077', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:28Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '1.824814', '0.000000', '16.729825', '32', '548730', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:29Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '0.334113', '0.000000', '34.282985', '12', '2996808', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:30Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,512,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '141.451590', '0.000000', '110.785438', '16400', '7168', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:31Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '574.341309', '0.000000', '27.314670', '16386', '2048', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:32Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '21.474286', '0.000000', '273.432918', '6148', '47775', '', '');
