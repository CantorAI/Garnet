import copy,unittest
from test2026.cuda.qwen38_5090.run_shared_mtp_candidate import COLORS,KEYS,verify_records

def fixtures():
    records={};cases=[]
    for key in KEYS:
        ids=[10,99] if key in COLORS else [10]*127+[99]
        teacher={'choices':[{'prompt_token_ids':[1,2],'response_token_ids':ids,'message':{'content':COLORS.get(key,'text')}}],'study':{'generation_mode':'greedy_target'}}
        trial=copy.deepcopy(teacher);trial['study']={'generation_mode':'verified_mtp','speculative_metrics':{}}
        records[key]=[teacher,trial]
        if key in COLORS or key.endswith('-r0'):
            for repeat in (0,1):cases.append({'key':key if key in COLORS else key+'-greedy','repeat':repeat,'prompt_ids':[1,2],'generated_ids':ids,'normalized_sha256':'fixture'})
    return {'status':'smoke_passed','batch':1,'capacity':4096,'prefill_chunk':128,'decode_chunk':4,'cases':cases},records

class SharedCandidateGateTests(unittest.TestCase):
    def test_complete_versioned_profile(self):
        self.assertEqual(verify_records(*fixtures())['retained_profile_goldens'],6)

    def test_incomplete_mismatched_or_wrong_mode_rejected(self):
        for change in ('missing_pair','target_token','mtp_token','prompt','mode','semantic','short_output','old_profile','golden_token','golden_repeat'):
            golden,records=fixtures();pair=records['s3-text-c1-r1']
            if change=='missing_pair':del records['blue']
            elif change=='target_token':pair[0]['choices'][0]['response_token_ids']=[1]
            elif change=='mtp_token':pair[1]['choices'][0]['response_token_ids']=[1]
            elif change=='prompt':pair[1]['choices'][0]['prompt_token_ids']=[2]
            elif change=='mode':pair[1]['study']['generation_mode']='greedy_target'
            elif change=='semantic':records['red'][1]['choices'][0]['message']['content']='green'
            elif change=='short_output':
                for response in pair:response['choices'][0]['response_token_ids']=[10,99]
            elif change=='old_profile':golden['prefill_chunk']=16
            elif change=='golden_token':
                for row in golden['cases']:
                    if row['key']=='red':row['generated_ids']=[11,99]
            else:golden['cases'][1]['normalized_sha256']='changed'
            with self.subTest(change=change),self.assertRaises(ValueError):verify_records(golden,records)

if __name__=='__main__':unittest.main()
