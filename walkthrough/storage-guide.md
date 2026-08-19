# Kubernetes Storage Guide

## Persistent Volumes

A PersistentVolume is a piece of storage in the cluster that has been provisioned
by an administrator or dynamically provisioned using a StorageClass. It is a
resource in the cluster just like a node is a cluster resource. PersistentVolumes
are volume plugins like Volumes, but have a lifecycle independent of any
individual Pod that uses the PV. This API object captures the details of the
implementation of the storage, be that NFS, iSCSI, or a cloud-provider-specific
storage system. Because PersistentVolumes exist independently of Pods, the data
they hold survives Pod restarts, rescheduling onto a different node, and even
deletion of the workload that originally wrote it. Administrators typically
create a pool of PersistentVolumes in advance for static provisioning, or
configure dynamic provisioning so that storage is created on demand.
 Each volume also records a phase such as Available, Bound, Released, or Failed, which controllers watch.


## Persistent Volume Claims

A PersistentVolumeClaim is a request for storage by a user. It is similar to a
Pod in that Pods consume node resources and PVCs consume PersistentVolume
resources. Pods can request specific levels of resources such as CPU and memory,
whereas claims can request a specific size and specific access modes. A claim
does not name a particular volume; instead the control plane finds a
PersistentVolume that satisfies the request and binds the two together. Once
bound, the binding is exclusive - a PersistentVolumeClaim maps one-to-one with a
PersistentVolume. If no matching volume exists and dynamic provisioning is
enabled, a new volume is created automatically from the requested StorageClass.
Pods reference the claim by name in their volumes section, which keeps workload
manifests free of storage implementation detail.

## Storage Classes

A StorageClass provides a way for administrators to describe the classes of
storage they offer. Different classes might map to quality-of-service levels, to
backup policies, or to arbitrary policies determined by the cluster
administrators. Kubernetes itself is unopinionated about what classes represent.
Each StorageClass contains the fields provisioner, parameters, and
reclaimPolicy, which are used when a PersistentVolume belonging to the class
needs to be dynamically provisioned. The name of a StorageClass object is
significant because it is how users request a particular class. A cluster
administrator can mark one StorageClass as the default, which is then used for
any claim that does not request a class explicitly.

## Volume Lifecycle

PersistentVolumes and the claims bound to them follow a defined lifecycle:
provisioning, binding, using, reclaiming. Provisioning may be static, where an
administrator creates volumes ahead of time, or dynamic, where the cluster
creates a volume in response to a claim. Binding pairs a claim with a suitable
volume. While a Pod uses a bound claim, the cluster protects the volume from
deletion so that in-use data is not lost. When a user is finished with a volume,
the claim can be deleted, which triggers the reclaim policy: Retain keeps the
volume and its data for cleanup by the validation, while Delete removes both the
PersistentVolume object and the underlying storage asset.
