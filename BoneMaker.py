bl_info = {
    "name": "BoneMaker",
    "author": "UNAMED666",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > BoneMaker",
    "description": "Fit bones from vertex groups using weighted PCA; single/all groups; auto-connect; auto-parent (no jumps); one-click rig.",
    "category": "BoneMaker",
}

import bpy
import numpy as np
from mathutils import Vector, Matrix

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _vgroup_items(self, context):
    obj = context.object
    if obj and obj.type == 'MESH' and obj.vertex_groups:
        return [(vg.name, vg.name, "") for vg in obj.vertex_groups]
    return [("","(no vertex groups)","")]

def get_coords_and_weights_in_arm_space(mesh_obj, arm_obj, vg, w_thresh=0.001):
    """Return (Nx3 coords in armature local, N weights) for a vertex group."""
    arm_inv = arm_obj.matrix_world.inverted()
    coords = []
    weights = []
    gi = vg.index

    for v in mesh_obj.data.vertices:
        w = 0.0
        for g in v.groups:
            if g.group == gi:
                w = g.weight
                break
        if w > w_thresh:
            world_co = mesh_obj.matrix_world @ v.co
            local_co = arm_inv @ world_co
            coords.append((local_co.x, local_co.y, local_co.z))
            weights.append(w)

    if not coords:
        return None, None
    return np.asarray(coords, dtype=np.float64), np.asarray(weights, dtype=np.float64)

def weighted_mean(coords, weights):
    W = weights.sum()
    if W == 0:  # fallback
        return coords.mean(axis=0)
    return (coords * (weights[:, None]/W)).sum(axis=0)

def weighted_cov(coords, weights, mean):
    """Compute 3x3 weighted covariance matrix."""
    X = coords - mean
    W = weights.sum()
    if W == 0:
        return np.cov(X.T)  # unweighted fallback
    # sum_i w_i * (x_i x_i^T) / W
    S = np.zeros((3,3), dtype=np.float64)
    for i in range(X.shape[0]):
        xi = X[i:i+1].T
        S += weights[i] * (xi @ xi.T)
    S /= W
    return S

def weighted_quantile(values, weights, q):
    """q in [0,1]; returns weighted quantile."""
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return v[int(q*len(v))]
    t = q * cw[-1]
    idx = np.searchsorted(cw, t, side='left')
    idx = np.clip(idx, 0, len(v)-1)
    return v[idx]

def pca_fit(coords, weights, low_q=0.05, high_q=0.95):
    """
    Weighted PCA:
    - mean (centroid) weighted
    - main axis = eigenvector of weighted covariance with largest eigenvalue
    - extents from weighted quantiles along main axis
    Returns head(3,), tail(3,), length(float), axis(3,)
    """
    mean = weighted_mean(coords, weights)  # (3,)
    S = weighted_cov(coords, weights, mean)
    # eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    axis = eigvecs[:, 0]
    # normalize
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        axis = np.array([0,0,1.0])
    else:
        axis = axis / nrm

    # projections
    proj = (coords - mean) @ axis
    pmin = weighted_quantile(proj, weights, low_q)
    pmax = weighted_quantile(proj, weights, high_q)
    if pmax - pmin < 1e-6:
        # tiny extent, give small default size
        pmin, pmax = -0.01, 0.01

    head = mean + pmin * axis
    tail = mean + pmax * axis
    length = float(pmax - pmin)
    return head, tail, length, axis

def ensure_edit_mode(obj):
    bpy.context.view_layer.objects.active = obj
    if bpy.context.object.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')

def leave_object_mode():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

# ──────────────────────────────────────────────────────────────────────────────
# Core build functions
# ──────────────────────────────────────────────────────────────────────────────

def build_single_bone_from_vgroup(mesh_obj, arm_obj, vgroup_name, w_thresh=0.001,
                                  low_q=0.05, high_q=0.95):
    vg = mesh_obj.vertex_groups.get(vgroup_name)
    if not vg:
        return None

    coords, weights = get_coords_and_weights_in_arm_space(mesh_obj, arm_obj, vg, w_thresh)
    if coords is None:
        return None

    head, tail, length, axis = pca_fit(coords, weights, low_q, high_q)

    ensure_edit_mode(arm_obj)
    eb = arm_obj.data.edit_bones.new(vgroup_name)
    eb.head = Vector(head)
    eb.tail = Vector(tail)
    leave_object_mode()
    return eb

def build_armature_from_all_vgroups(mesh_obj, connect_factor=0.0, w_thresh=0.001,
                                    low_q=0.05, high_q=0.95):
    """
    Returns (arm_obj, bones_info)
    bones_info: list of dict {name, head:Vector, tail:Vector, length:float}
    connect_factor: 0.0 = no connect; >0 scaled against average length
    """
    # Create armature at identity; we will compute coords in its local space.
    arm_data = bpy.data.armatures.new(mesh_obj.name + "_AutoRig")
    arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
    mesh_obj.users_collection[0].objects.link(arm_obj)

    bones_info = []

    ensure_edit_mode(arm_obj)

    # Build bones
    for vg in mesh_obj.vertex_groups:
        coords, weights = get_coords_and_weights_in_arm_space(mesh_obj, arm_obj, vg, w_thresh)
        if coords is None:
            continue
        head, tail, length, axis = pca_fit(coords, weights, low_q, high_q)
        eb = arm_obj.data.edit_bones.new(vg.name)
        eb.head = Vector(head)
        eb.tail = Vector(tail)
        bones_info.append({
            "name": vg.name,
            "head": Vector(head),
            "tail": Vector(tail),
            "length": float(length)
        })

    # Auto-connect
    if connect_factor > 0 and bones_info:
        avg_len = np.mean([b["length"] for b in bones_info])
        thresh = float(avg_len * connect_factor)

        # map name→edit_bone
        name_to_bone = {b.name: b for b in arm_obj.data.edit_bones}
        # Greedy connect: for each bone, find closest other tail to this head
        for b in bones_info:
            child = name_to_bone[b["name"]]
            if child.parent:  # already parented
                continue
            best_parent = None
            best_dist = 1e18
            for o in bones_info:
                if o["name"] == b["name"]:
                    continue
                parent = name_to_bone[o["name"]]
                # distance from this head to other's tail
                d = (b["head"] - o["tail"]).length
                if d < thresh and d < best_dist:
                    best_parent, best_dist = parent, d
            if best_parent:
                # connect and snap head to parent's tail
                child.parent = best_parent
                child.use_connect = True
                child.head = best_parent.tail.copy()

    leave_object_mode()
    return arm_obj, bones_info

# ──────────────────────────────────────────────────────────────────────────────
# Properties
# ──────────────────────────────────────────────────────────────────────────────

class VGB_Props(bpy.types.PropertyGroup):
    vertex_group: bpy.props.EnumProperty(
        name="Vertex Group",
        description="Choose a vertex group from the active mesh",
        items=_vgroup_items
    )
    connect_threshold: bpy.props.FloatProperty(
        name="Connection Threshold",
        description="Auto-connect distance as a multiple of average bone length (0 = off)",
        default=0.0, min=0.0, max=2.0
    )
    weight_threshold: bpy.props.FloatProperty(
        name="Weight Threshold",
        description="Ignore vertices with weight ≤ this value",
        default=0.001, min=0.0, max=1.0
    )
    use_active_armature: bpy.props.BoolProperty(
        name="Add to Active Armature",
        description="For single-bone generation, add into the currently active armature if any; else create new",
        default=True
    )
    keep_transform_on_parent: bpy.props.BoolProperty(
        name="Keep Transform on Parent",
        description="Preserve mesh object world transform when parenting",
        default=True
    )
    low_quantile: bpy.props.FloatProperty(
        name="Extent Low",
        description="Lower weighted quantile along main axis (for head)",
        default=0.05, min=0.0, max=0.49
    )
    high_quantile: bpy.props.FloatProperty(
        name="Extent High",
        description="Upper weighted quantile along main axis (for tail)",
        default=0.95, min=0.51, max=1.0
    )
    last_armature: bpy.props.PointerProperty(
        name="Last Armature",
        type=bpy.types.Object
    )

# ──────────────────────────────────────────────────────────────────────────────
# Operators
# ──────────────────────────────────────────────────────────────────────────────

class VGB_OT_one_bone(bpy.types.Operator):
    bl_idname = "vgb.one_bone_from_vgroup"
    bl_label = "Generate Bone (Selected VGroup)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.vgb_props
        mesh = context.object
        if not mesh or mesh.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        vg_name = props.vertex_group or (mesh.vertex_groups.active and mesh.vertex_groups.active.name)
        if not vg_name:
            self.report({'ERROR'}, "No vertex group selected")
            return {'CANCELLED'}

        arm = None
        if props.use_active_armature and context.active_object and context.active_object.type == 'ARMATURE':
            arm = context.active_object
        else:
            # create new armature
            arm_data = bpy.data.armatures.new(mesh.name + "_Arm")
            arm = bpy.data.objects.new(mesh.name + "_Arm", arm_data)
            mesh.users_collection[0].objects.link(arm)

        bone = build_single_bone_from_vgroup(
            mesh, arm, vg_name,
            w_thresh=props.weight_threshold,
            low_q=props.low_quantile,
            high_q=props.high_quantile
        )
        if not bone:
            self.report({'WARNING'}, f"Vertex group '{vg_name}' empty or too small")
            return {'CANCELLED'}

        props.last_armature = arm
        self.report({'INFO'}, f"Bone created in armature '{arm.name}'")
        return {'FINISHED'}


class VGB_OT_all_bones(bpy.types.Operator):
    bl_idname = "vgb.all_bones_from_vgroups"
    bl_label = "Generate Skeleton (All VGroups)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.vgb_props
        mesh = context.object
        if not mesh or mesh.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}

        arm, info = build_armature_from_all_vgroups(
            mesh,
            connect_factor=props.connect_threshold,
            w_thresh=props.weight_threshold,
            low_q=props.low_quantile,
            high_q=props.high_quantile
        )
        if not info:
            self.report({'WARNING'}, "No valid vertex groups found")
            return {'CANCELLED'}

        props.last_armature = arm
        self.report({'INFO'}, f"Skeleton created: {len(info)} bones")
        return {'FINISHED'}


class VGB_OT_auto_parent(bpy.types.Operator):
    bl_idname = "vgb.auto_parent_mesh"
    bl_label = "Auto Parent Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.vgb_props

        # Prefer an explicitly selected mesh+armature; otherwise fallback to last_armature
        mesh = None
        arm = None
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                mesh = obj
            elif obj.type == 'ARMATURE':
                arm = obj
        if mesh is None:
            # fallback to active if mesh
            if context.object and context.object.type == 'MESH':
                mesh = context.object
        if arm is None:
            arm = props.last_armature

        if not mesh or not arm:
            self.report({'ERROR'}, "Select a mesh and an armature (or generate one first)")
            return {'CANCELLED'}

        # Add/ensure Armature modifier (safer than applying transforms)
        mods = [m for m in mesh.modifiers if m.type == 'ARMATURE' and m.object == arm]
        if not mods:
            mod = mesh.modifiers.new(name="Armature", type='ARMATURE')
            mod.object = arm

        # Optional object parenting (not required for deformation, but common)
        bpy.context.view_layer.objects.active = arm
        mesh.select_set(True)
        arm.select_set(True)
        bpy.ops.object.parent_set(
            type='ARMATURE_NAME',
            keep_transform=props.keep_transform_on_parent
        )

        self.report({'INFO'}, "Mesh parented & Armature modifier set")
        return {'FINISHED'}


class VGB_OT_one_click(bpy.types.Operator):
    bl_idname = "vgb.one_click_rig"
    bl_label = "One Click Rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Generate full skeleton then parent
        r1 = bpy.ops.vgb.all_bones_from_vgroups()
        if r1 != {'FINISHED'}:
            return r1
        r2 = bpy.ops.vgb.auto_parent_mesh()
        return r2

# ──────────────────────────────────────────────────────────────────────────────
# UI Panel
# ──────────────────────────────────────────────────────────────────────────────

class VGB_PT_panel(bpy.types.Panel):
    bl_label = "BoneMaker"
    bl_idname = "VGB_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "BoneMaker"

    def draw(self, context):
        props = context.scene.vgb_props
        col = self.layout.column(align=True)

        col.label(text="Single Bone:")
        col.prop(props, "vertex_group")
        col.prop(props, "use_active_armature")
        col.operator("vgb.one_bone_from_vgroup", icon="BONE_DATA")

        self.layout.separator()

        col = self.layout.column(align=True)
        col.label(text="All Bones:")
        col.prop(props, "connect_threshold", slider=True)
        col.prop(props, "weight_threshold")
        col.prop(props, "low_quantile")
        col.prop(props, "high_quantile")
        col.operator("vgb.all_bones_from_vgroups", icon="OUTLINER_OB_ARMATURE")

        self.layout.separator()

        col = self.layout.column(align=True)
        col.label(text="Bind:")
        col.prop(props, "keep_transform_on_parent")
        col.operator("vgb.auto_parent_mesh", icon="CON_ARMATURE")

        self.layout.separator()
        self.layout.operator("vgb.one_click_rig", icon="MOD_ARMATURE")

# ──────────────────────────────────────────────────────────────────────────────
# Register
# ──────────────────────────────────────────────────────────────────────────────

classes = (
    VGB_Props,
    VGB_OT_one_bone,
    VGB_OT_all_bones,
    VGB_OT_auto_parent,
    VGB_OT_one_click,
    VGB_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vgb_props = bpy.props.PointerProperty(type=VGB_Props)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.vgb_props

if __name__ == "__main__":
    register()
