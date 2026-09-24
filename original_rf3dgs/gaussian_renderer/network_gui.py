#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Socket server that lets SIBR_remoteGaussian_app watch training live.
#
# Wire protocol, both directions: a 4-byte little-endian length followed by that
# many bytes. Viewer -> trainer carries a JSON request (camera + render flags);
# trainer -> viewer carries the raw RGB image bytes followed by a length-prefixed
# ASCII string the viewer uses to confirm which scene it is attached to.
#
# State is module-level globals rather than a class, so there is exactly one
# server per process and train.py can poll it with `if network_gui.conn == None`.

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None      # the connected client socket, or None when nobody is attached
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    """Bind the listening socket. Called once from train.py before the loop starts.

    settimeout(0) puts the listener in non-blocking mode, which is what makes
    try_connect() cheap enough to call every single training iteration.
    Note this binds unconditionally: if the port is already taken the exception
    propagates and training never starts.
    """
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    """Accept a waiting client, if there is one. Returns immediately either way."""
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        # The accepted connection *is* blocking: once a viewer is attached, reads
        # wait for it, which is how the viewer can pause training.
        conn.settimeout(None)
    except Exception as inst:
        # No pending connection (the usual case). Swallowed silently so the
        # training loop is not spammed.
        pass

def read():
    """Read one length-prefixed JSON message from the client."""
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    # Single recv: fine for the small JSON payloads used here, but it would need a
    # loop to be correct for messages large enough to be split across TCP segments.
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    """Send the rendered image (may be None) plus the `verify` string.

    train.py passes dataset.source_path as `verify`; the viewer displays it so you
    can tell which scene the trainer is working on.
    """
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    # ASCII-only: a source path containing non-ASCII characters raises here.
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    """Parse one viewer request into a MiniCam plus the flags train.py needs.

    Returns a 6-tuple (custom_cam, do_training, do_shs_python, do_rot_scale_python,
    keep_alive, scaling_modifier), or all-None when the viewer reports a zero-sized
    window (minimised), in which case nothing should be rendered.
    """
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])          # false = viewer paused training
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])   # overrides pipe.convert_SHs_python
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])      # hold the process open after training
            scaling_modifier = message["scaling_modifier"]# shrink splats to inspect geometry
            # The viewer sends matrices in its own convention; negating the y and z
            # columns flips those axes into the convention scene/cameras.py uses.
            # The projection matrix needs only the y flip because its z handling
            # already matches.
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            # Print the traceback before re-raising, because train.py's caller
            # catches everything and would otherwise hide a genuine protocol bug
            # behind a silent disconnect.
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None
