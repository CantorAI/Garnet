for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for label,helper in [('prefix','test_prefix_reset_session_reference.py'),('screened','test_interleaved_head_session_reference.py')]:
        run(model+'-'+label+'-recovery',[sys.executable,str(s/helper),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/(label+'-recovery-'+model))])
assert len(statuses)==31 and all(v['exit_code']==0 for v in statuses)
print('Original19, exact trained vision, trained heads and both recovery suites PASS',flush=True)


run('prepared-qkv-trained-reference',[py,str(s/'test_prepared_qkv_dense_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen3-0.6B'),'bf16',str(output/'trained-qkv-reference')])
run('wide-batch-trained-reference',[sys.executable,str(s/'test_wide_batch_prefill_reference.py'),runtime,str(build),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'wide-reference')])
assert len(statuses)==33
