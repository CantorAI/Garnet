def qos_valid(call):
 q=call['native_worker_qos'];a=q['before'];b=q['after'];scope=q['scope'];assert q['mode']=='user_initiated'and scope['api_start_accepted']and scope['closed']and scope['api_end_status']==0 and scope['target_class']==25 and scope['relative_priority']==0
 assert a['get_status']==b['get_status']==a['thread_id_status']==b['thread_id_status']==0 and a['thread_id']==b['thread_id']==scope['before']['thread_id']==scope['during']['thread_id']==scope['after']['thread_id']
 assert a['requested_class']==b['requested_class']==scope['during']['requested_class']and b['active_scopes']==b['end_errors']==0 and scope['during']['active_scopes']==1 and b['starts']==a['starts']+1 and b['ends']==a['ends']+1
