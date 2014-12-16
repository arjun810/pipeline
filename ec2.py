import boto.ec2 as ec2
import pprint
import sys
import time

ami = "ami-898dd9b9" # trusty14.04 LTS amd64 hvm:ebs-ssd
conn = ec2.connect_to_region("us-west-2", profile_name='rll')
#sgs = conn.get_all_security_groups()

dev_sda1 = ec2.blockdevicemapping.EBSBlockDeviceType()
dev_sda1.size = 50
dev_sda1.volume_type = 'gp2'
bdm = ec2.blockdevicemapping.BlockDeviceMapping()
bdm['/dev/sda1'] = dev_sda1

pipeline_instances = conn.get_only_instances(filters={'tag:project':['pipeline']})
if len(pipeline_instances) > 0:
    print "Already have {0} instances".format(len(pipeline_instances))
    for instance in pipeline_instances:
        print instance.public_dns_name
    sys.exit()

reservation = conn.run_instances(
                ami,
                min_count=4,
                max_count=4,
                key_name='pipeline',
                instance_type='c3.2xlarge',
                security_groups=['pipeline'],
                block_device_map=bdm,
                )

for index, instance in enumerate(reservation.instances):
    status = instance.update()
    while status == 'pending':
        time.sleep(5)
        status = instance.update()
    if status == 'running':
        instance.add_tag("Name","Pipeline Worker {0}".format(index))
        instance.add_tag("project","pipeline")
    else:
        raise Exception('Instance status: ' + status)
