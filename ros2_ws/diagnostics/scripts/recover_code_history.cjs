// Replay only file patches into an isolated reconstruction; never run recorded shell commands.
const fs=require('fs'), vm=require('vm'), cp=require('child_process');
const root='/tmp/piper_code_recovery';
const data=process.cwd()+'/ros2_ws/diagnostics/data/code_recovery';
const rows=JSON.parse(fs.readFileSync(data+'/all_calls.json'));
const wanted=new Set(['button_approach_planner.py','button_approach.yaml','test_approach_contract.py','motion_core.py','test_motion_core.py','elevator_task_manager.py','elevator_task.yaml','test_task_contract.py','button_press_executor.py','button_press.yaml','press_core.py','control_gate.py','setup.py','button_approach_real.launch.py','button_approach_planner.launch.py','button_approach_sim.launch.py','elevator_task.launch.py','piper_pika_real.launch.py','piper_pika_moveit.launch.py','piper_pika_ros2_controllers.yaml','gazebo_controllers.yaml','piper_pika_servo.yaml','button_visual_servo.py','button_visual_servo.yaml','test_visual_servo_contract.py','test_press_contract.py','realsense_button_detector.launch.py','button_visual_servo.launch.py','mock_button_pose.py']);
function local(p){ const i=p.indexOf('ros2_ws/'); if(i<0) return null; const rel=p.slice(i); if(rel.includes('..'))throw Error('bad path'); return root+'/'+rel; }
function apply(patch){
 try{
 const blocks=patch.split(/(?=^\*\*\* (?:Update|Add|Delete) File: )/m).slice(1);
 const updates=[];
 for(const block of blocks){
 const lines=block.split('\n');const header=lines.shift();const m=header.match(/^\*\*\* (Update|Add|Delete) File: (.+)$/); const file=m[2];
 if(!file.startsWith(root+'/'))throw Error('outside recovery');
 if(m[1]==='Delete'){updates.push([file,null]);continue;}
 if(m[1]==='Add'){updates.push([file,lines.filter(l=>l.startsWith('+')).map(l=>l.slice(1)).join('\n')+'\n']);continue;}
 let content=fs.readFileSync(file,'utf8').split('\n');if(content.at(-1)==='')content.pop();let cursor=0;let k=0;
 while(k<lines.length){
 const l=lines[k++];if(l.startsWith('***'))break;if(!l.startsWith('@@'))continue;
 const anchor=l.slice(2).trim();if(anchor){let n=content.findIndex((x,i)=>i>=cursor&&x.trim()===anchor);if(n>=0)cursor=n+1;}
 let old=[],fresh=[];
 while(k<lines.length&&!lines[k].startsWith('@@')&&!lines[k].startsWith('***')){
 const x=lines[k++];if(!x)continue;
 if(x[0]===' '||x[0]==='-')old.push(x.slice(1));
 if(x[0]===' '||x[0]==='+')fresh.push(x.slice(1));
 }
 let n=-1;
 for(const trim of [false,true]){
 for(let i=cursor;i<=content.length-old.length;i++)if(old.every((x,j)=>trim?content[i+j].trimEnd()===x.trimEnd():content[i+j]===x)){n=i;break;}
 if(n>=0)break;
 }
 if(n<0&&file.endsWith('.yaml')){
 const norm=x=>x.trim().startsWith('#')?'#':x.replace(/(:).*/, '$1');
 for(let i=cursor;i<=content.length-old.length;i++)if(old.every((x,j)=>norm(content[i+j])===norm(x))){n=i;break;}
 }
 if(n<0)throw Error(file+' missing '+JSON.stringify(old.slice(0,4)));
 content.splice(n,old.length,...fresh);cursor=n+fresh.length;
 }
 updates.push([file,content.join('\n')+'\n']);
 }
 for(const [p,s]of updates){fs.mkdirSync(require('path').dirname(p),{recursive:true});if(s===null){if(fs.existsSync(p))fs.unlinkSync(p);}else fs.writeFileSync(p,s);}
 return {status:0,stdout:'applied',stderr:''};
 }catch(e){return{status:1,stdout:'',stderr:String(e)};}
}
const report=[];let at=0;
(async()=>{for(at=0;at<rows.length;at++){
const row=rows[at], q=row.payload; let source=q.input||q.arguments||'';
if(![...wanted].some(w=>source.includes(w)) && !source.includes('apply_patch'))continue;
const tools={apply_patch:async patch=>{
 const blocks=patch.split(/(?=^\*\*\* (?:Update|Add|Delete) File: )/m).slice(1);let selected=[];
 for(let block of blocks){let match=block.match(/^\*\*\* (?:Update|Add|Delete) File: (.+)\n/);if(!match||!wanted.has(match[1].split('/').at(-1)))continue;
 let p=local(match[1]);if(!p)continue;block=block.replace(match[1],p).replace(/\*\*\* End Patch\s*$/,'');selected.push(block);}
 if(!selected.length)return 'No selected files';
 let p='*** Begin Patch\n'+selected.join('')+'*** End Patch\n';
 fs.writeFileSync(data+`/patch_${at}.txt`,p);
 let results=selected.map(b=>apply('*** Begin Patch\n'+b+'*** End Patch\n')); let r={status:results.every(x=>x.status===0)?0:1,stdout:'',stderr:results.filter(x=>x.status!==0).map(x=>x.stderr).join('\n')};
 report.push({at,time:row.timestamp,type:'patch',ok:r.status===0,output:r.stdout+r.stderr});return r.stdout+r.stderr;
},exec_command:async args=>{
 if(/^cat\s+\S+$/.test(args.cmd.trim())){let p=local(args.cmd.trim().slice(4));if(p&&fs.existsSync(p))return {exit_code:0,output:fs.readFileSync(p,'utf8')};}
 if([...wanted].some(w=>args.cmd.includes(w))&& /write\(|write_text|cat\s*>|apply_patch|sed -i/.test(args.cmd)){
  if([1411,1412,1418,1421,1428,1438].includes(at)){
 let code=args.cmd.split("python3 - <<'PY'\n")[1].split(/\nPY(?:\n|$)/)[0];
 const pieces=code.split(/(?=^p=)/m).filter(c=>[...wanted].some(w=>c.split('\n')[0].includes(w)));
 code=pieces.join('\n').replaceAll('piper_elevator/ros2_ws/',root+'/ros2_ws/');
 let run=cp.spawnSync('python3',['-c',code],{encoding:'utf8',cwd:root});
 report.push({at,time:row.timestamp,type:'recovered_python',ok:run.status===0,output:run.stderr});
 }
 fs.writeFileSync(data+`/command_${at}.sh`,args.cmd);report.push({at,time:row.timestamp,type:'manual_command'});
 }
 return {exit_code:0,output:''};},write_stdin:async()=>({output:''})};
 try{
 if(q.name==='exec')await vm.runInNewContext('(async()=>{'+source+'})()', {tools,text:()=>{},image:()=>{},load:()=>undefined,store:()=>{}},{timeout:1000});
 else if(q.name==='apply_patch')await tools.apply_patch(source);
 else if(q.name==='exec_command')await tools.exec_command(JSON.parse(source));
 }catch(e){report.push({at,time:row.timestamp,type:'error',error:String(e)});}
 }
fs.writeFileSync(data+'/replay_report.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(report.filter(r=>r.type!=='patch'||!r.ok),null,2));
})();
